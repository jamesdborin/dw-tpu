# SPDX-License-Identifier: Apache-2.0
"""P/D KV connector for DeepSeek V4 (hybrid, overlaid KV cache arrays).

Selected with ``--kv-transfer-config '{"kv_connector": "TPUConnectorDSV4",
"kv_connector_module_path": "tpu_inference.distributed.tpu_connector_dsv4",
"kv_role": "kv_producer"|"kv_consumer"}'``. Nothing else changes when it is
not selected.

Why TPUConnectorHMA is not enough for DSV4:

1. DSV4's cache groups (compressed CSA/HCA MLA caches, indexer k_cache, SWA
   caches, compressor state caches) are *overlaid*: several layers from
   different groups share one JAX array (a block ID is owned by one group at
   a time), and every CSA layer has a companion ``<layer>_rope`` array that is
   in no group. TPUConnectorHMA maps every array to exactly one group, so it
   transfers some arrays with the wrong group's block IDs and the rope arrays
   with group 0's. Here a transfer *unit* is an ``(array, group)`` pair, and
   the rope array is a unit of its layer's group.
2. Sliding-window groups (SWA, compressor states) only need the trailing
   window; the rest of their block list is (or becomes) the null block.
   Only the tail window is transferred, and null blocks are never written.
3. With a local prefix hit on D, the HMA connector still pulls (and
   rewrites) every block. Here P registers the full-attention groups in
   chunks of ``DSV4_PD_CHUNK_TOKENS`` tokens (default 1024) and D pulls only
   the chunks that reach past its local hit, so cache-aware decode routing
   saves transfer bytes. Blocks D already holds are never overwritten.
4. D sent the "pull done" notification to its *own* side-channel port, so
   with several P instances on one host it notified the wrong one. P now
   puts its side-channel port into ``kv_transfer_params``.

Transfer plan (computed identically on P and D from the prompt length L):
  full group g (block size b):   logical blocks [0, ceil(L/b)), cut into
                                 chunks of C/b blocks -> entry k
  sliding group g (window w):    logical blocks [max(0, L-w-b)//b,
                                 ceil(L/b)) -> one tail entry
Entry ids: ``(uuid << 12) | k`` for chunk k, ``(uuid << 12) | 4095`` for the
tail.
"""

import datetime
import functools
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole)
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        SlidingWindowSpec)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import zmq

import tpu_inference.distributed.utils as dist_utils
# isort: off
from tpu_inference.distributed.tpu_connector import (LoadMeta, SendMeta,
                                                     get_uuid)
from tpu_inference.distributed.tpu_connector_hma import (
    TPUConnectorHMA, TPUConnectorHMAScheduler, TPUConnectorHMAWorker)
# isort: on
from tpu_inference.distributed.transfer_stats import TransferStats
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

ReqId = str

_ENTRY_BITS = 12
_TAIL_ENTRY = (1 << _ENTRY_BITS) - 1
_ROPE_SUFFIX = "_rope"


def get_chunk_tokens() -> int:
    return int(os.getenv("DSV4_PD_CHUNK_TOKENS", "1024"))


def entry_uuid(uuid: int, entry: int) -> int:
    return (uuid << _ENTRY_BITS) | entry


@dataclass
class SendMetaDSV4(SendMeta):
    num_tokens: int = 0
    chunk_tokens: int = 0


@dataclass
class LoadMetaDSV4(LoadMeta):
    num_tokens: int = 0
    chunk_tokens: int = 0
    # Tokens D already holds in its own prefix cache (multiple of the hash
    # block size); blocks below it are neither pulled nor written.
    local_hit_tokens: int = 0
    remote_side_port: Optional[int] = None


@dataclass(frozen=True)
class GroupInfo:
    block_size: int
    # None for full-attention groups.
    sliding_window: Optional[int]


def _group_info(spec) -> GroupInfo:
    specs = list(spec.kv_cache_specs.values()) if hasattr(
        spec, "kv_cache_specs") else [spec]
    if all(isinstance(s, SlidingWindowSpec) for s in specs):
        window = max(s.sliding_window for s in specs)
    elif all(isinstance(s, FullAttentionSpec) for s in specs):
        window = None
    else:
        raise NotImplementedError(
            f"TPUConnectorDSV4: unsupported cache group specs {specs}")
    return GroupInfo(block_size=spec.block_size, sliding_window=window)


def plan_entries(num_tokens: int, chunk_tokens: int,
                 groups: list[GroupInfo]) -> dict[int, dict[int, list[int]]]:
    """Logical block indices to transfer, per entry and per group.

    Returns ``{entry: {group_id: [logical block index, ...]}}``; chunk entries
    hold the full-attention groups, the tail entry the sliding groups.
    """
    entries: dict[int, dict[int, list[int]]] = {}
    for gid, g in enumerate(groups):
        nblocks = -(-num_tokens // g.block_size)
        if g.sliding_window is None:
            assert chunk_tokens % g.block_size == 0, (
                f"chunk {chunk_tokens} not a multiple of block size "
                f"{g.block_size}")
            per_chunk = chunk_tokens // g.block_size
            for k in range(-(-nblocks // per_chunk)):
                entries.setdefault(k, {})[gid] = list(
                    range(k * per_chunk, min((k + 1) * per_chunk, nblocks)))
        else:
            first = max(0, num_tokens - g.sliding_window -
                        g.block_size) // g.block_size
            entries.setdefault(_TAIL_ENTRY, {})[gid] = list(
                range(first, nblocks))
    assert len([e for e in entries if e != _TAIL_ENTRY]) < _TAIL_ENTRY
    return entries


def entries_needed(num_tokens: int, chunk_tokens: int, local_hit_tokens: int,
                   entries: dict[int, dict[int, list[int]]]) -> list[int]:
    """Entries D has to pull: chunks reaching past its hit, and the tail."""
    needed = []
    for e in sorted(entries):
        if e == _TAIL_ENTRY or (e + 1) * chunk_tokens > local_hit_tokens:
            needed.append(e)
    return needed


class TPUConnectorDSV4(TPUConnectorHMA):

    def __init__(self,
                 vllm_config: VllmConfig,
                 role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig | None" = None):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPUConnectorDSV4Scheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = TPUConnectorDSV4Worker(vllm_config)

    def get_loaded_kv_cache_group_ids(self, request: "Request"):
        # The load is the request's own blocks for every group.
        return self._kv_cache_config.transfer_group_ids


class TPUConnectorDSV4Scheduler(TPUConnectorHMAScheduler):

    def __init__(self, vllm_config: "VllmConfig"):
        super().__init__(vllm_config)
        self.side_channel_port = int(dist_utils.get_side_channel_port())
        self.chunk_tokens = get_chunk_tokens()
        # D: local prefix-hit tokens seen in get_num_new_matched_tokens.
        self._local_hit: dict[ReqId, int] = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer or not request.kv_transfer_params:
            return 0, False
        # P sends the KV of its prompt, which is this prompt or a prefix of it
        # (the router may prefill only an aligned prefix on P); D computes
        # the rest itself.
        target = request.kv_transfer_params.get(
            "num_tokens", len(request.prompt_token_ids))
        count = max(target - num_computed_tokens, 0)
        is_async = count > 0
        if count > 0:
            self._local_hit[request.request_id] = num_computed_tokens
        return count, is_async

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return
        params = request.kv_transfer_params
        common = dict(uuid=params["uuid"],
                      remote_host=params["remote_host"],
                      remote_port=params["remote_port"],
                      num_tokens=params.get("num_tokens", 0),
                      chunk_tokens=params.get("chunk_tokens", 0),
                      remote_side_port=params.get("remote_side_port"))
        if num_external_tokens > 0:
            local_block_ids = [list(g) for g in blocks.get_block_ids()]
            self.reqs_to_load[request.request_id] = LoadMetaDSV4(
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                local_hit_tokens=self._local_hit.pop(request.request_id, 0),
                **common)
        else:
            self._local_hit.pop(request.request_id, None)
            # The pull has finished (or was not needed): the worker writes
            # the pulled KV into the blocks the request holds now, which the
            # forward pass of this step reads.
            self.reqs_to_load[request.request_id] = LoadMetaDSV4(
                local_block_ids=[list(g) for g in blocks.get_block_ids()],
                remote_block_ids=None,
                **common)
        meta = self.reqs_to_load[request.request_id]
        logger.info(f"TPUConnectorDSV4 Decode --> load queued | "
                    f"req_id={request.request_id} | uuid={meta.uuid} | "
                    f"pull={meta.remote_block_ids is not None} | "
                    f"num_tokens={meta.num_tokens} | "
                    f"local_hit_tokens={meta.local_hit_tokens}")

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self.is_producer:
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None
        num_tokens = len(request.prompt_token_ids)
        computed_per_group = [list(g) for g in block_ids]
        if num_tokens == 0 or not any(computed_per_group):
            return False, {}
        uuid = get_uuid()
        expiration_time = time.perf_counter(
        ) + dist_utils.get_p2p_wait_pull_timeout()
        self.reqs_to_send[request.request_id] = SendMetaDSV4(
            uuid=uuid,
            local_block_ids=computed_per_group,
            expiration_time=expiration_time,
            num_tokens=num_tokens,
            chunk_tokens=self.chunk_tokens)
        kv_transfer_params = dict(uuid=uuid,
                                  remote_block_ids=computed_per_group,
                                  remote_host=self.kv_ip,
                                  remote_port=self.kv_port,
                                  remote_side_port=self.side_channel_port,
                                  num_tokens=num_tokens,
                                  chunk_tokens=self.chunk_tokens)
        logger.info(f"TPUConnectorDSV4 Prefill --> send queued | "
                    f"req_id={request.request_id} | uuid={uuid} | "
                    f"num_tokens={num_tokens}")
        return True, kv_transfer_params


class TPUConnectorDSV4Worker(TPUConnectorHMAWorker):

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        # P: entries registered per uuid, dropped together on release.
        self._send_entries: dict[int, list[int]] = {}

    def register_runner(self, runner: TPUModelRunner):
        self.node_id = runner.topology_order_id
        self.runner = runner
        self.mesh = runner.mesh
        role = 'Prefill' if self.is_producer else 'Decode'
        self.stats = TransferStats(
            log_prefix=f"TPUConnectorDSV4 Worker {self.node_id} {role}")
        self.host_kv_pool = None
        assert not dist_utils.get_enable_d2h_transfer(), (
            "TPUConnectorDSV4 does not implement the D2H host pool")

        groups = runner.kv_cache_config.kv_cache_groups
        self.groups = [_group_info(g.kv_cache_spec) for g in groups]
        index = runner.layer_name_to_kvcache_index
        # Transfer units: (array index, group id); rope companions follow
        # their layer's group.
        units: list[tuple[int, int]] = []
        for gid, group in enumerate(groups):
            for name in group.layer_names:
                assert name in index, f"{name} has no kv cache array"
                units.append((index[name], gid))
                if name + _ROPE_SUFFIX in index:
                    units.append((index[name + _ROPE_SUFFIX], gid))
        assert len(set(units)) == len(units), f"duplicate units {units}"
        covered = {a for a, _ in units}
        missing = set(range(len(runner.kv_caches))) - covered
        assert not missing, f"kv cache arrays in no group: {sorted(missing)}"
        self.units = units
        self.units_by_group: dict[int, list[int]] = {}
        for u, (_, gid) in enumerate(units):
            self.units_by_group.setdefault(gid, []).append(u)
        self.null_block_id = 0
        # Transfer layout: "single" moves one copy through the first device of
        # the instance (KV is replicated over the TP devices, or sharded over
        # attention-DP ranks; the scatter on D redistributes it over ICI),
        # "replicated" moves the array replicated over all devices.
        self.xfer_layout = os.getenv("DSV4_PD_XFER_LAYOUT", "single")
        assert self.xfer_layout in ("single", "replicated"), self.xfer_layout
        cache_sharding = runner.kv_caches[0].sharding
        self.insert_sharding = _replicated(cache_sharding)
        if self.xfer_layout == "single":
            first = min(cache_sharding.device_set, key=lambda d: d.id)
            self.xfer_sharding = jax.sharding.SingleDeviceSharding(first)
        else:
            self.xfer_sharding = self.insert_sharding
        self._maybe_start_p2p_server()
        logger.info(
            f"TPUConnectorDSV4 Worker {self.node_id} {role} --> init | "
            f"ip={self.host_ip} | kv_transfer_port={self.kv_transfer_port} | "
            f"side_channel_port={self.side_channel_port} | "
            f"num_arrays={len(runner.kv_caches)} | groups={self.groups} | "
            f"units={self.units} | xfer_layout={self.xfer_layout} | "
            f"array_shapes={[tuple(a.shape) for a in runner.kv_caches]}")

    # ------------------------------------------------------------ helpers
    def _entry_units(self, plan: dict[int, list[int]]) -> list[tuple[int, int]]:
        """(unit index, group id) in transfer order for one entry."""
        return [(u, gid) for gid in sorted(plan)
                for u in self.units_by_group.get(gid, [])]

    def _entry_specs(self, plan: dict[int, list[int]]):
        specs = []
        for u, gid in self._entry_units(plan):
            arr = self.runner.kv_caches[self.units[u][0]]
            shape = list(arr.shape)
            shape[0] = len(plan[gid])
            specs.append(
                jax.ShapeDtypeStruct(shape,
                                     arr.dtype,
                                     sharding=self.xfer_sharding))
        return specs

    # --------------------------------------------------------------- P side
    def _prepare_kv_and_wait(self, req_id: str, req_meta: SendMeta):
        assert isinstance(req_meta, SendMetaDSV4), req_meta
        t0 = time.perf_counter()
        plan = plan_entries(req_meta.num_tokens, req_meta.chunk_tokens,
                            self.groups)
        block_ids = req_meta.local_block_ids
        entry_kv: dict[int, list[jax.Array]] = {}
        nbytes = 0
        for e, eplan in plan.items():
            arrays, idx = [], []
            for u, gid in self._entry_units(eplan):
                ids = block_ids[gid]
                assert len(ids) >= len(eplan[gid]) and (
                    not eplan[gid] or eplan[gid][-1] < len(ids)), (
                        f"group {gid}: {len(ids)} blocks, plan "
                        f"{eplan[gid][:1]}..{eplan[gid][-1:]}")
                arrays.append(self.runner.kv_caches[self.units[u][0]])
                idx.append(np.asarray([ids[i] for i in eplan[gid]],
                                      dtype=np.int32))
            kv = _gather(tuple(arrays), tuple(jnp.asarray(i) for i in idx))
            kv = [jax.device_put(k, self.xfer_sharding) for k in kv]
            entry_kv[e] = kv
            nbytes += sum(k.nbytes for k in kv)
            if _DEBUG:
                first = [_checksum(k) for k in kv]
                time.sleep(1.0)
                again = _gather(tuple(self.runner.kv_caches[self.units[u][0]]
                                      for u, _ in self._entry_units(eplan)),
                                tuple(jnp.asarray(i) for i in idx))
                logger.info("PDDEBUG send " + json.dumps(
                    dict(uuid=req_meta.uuid, entry=e,
                         units=self._entry_units(eplan),
                         ids={g: [block_ids[g][i] for i in eplan[g]]
                              for g in eplan},
                         checksum=first,
                         checksum_after_1s=[_checksum(k) for k in again])))
        t1 = time.perf_counter()
        with self._pull_done_lock:
            self.reqs_wait_pull[req_id] = [
                entry_kv, req_meta.expiration_time, -1
            ]
            self.kv_pull_uuid_to_req_id_map[req_meta.uuid] = req_id
            self._send_entries[req_meta.uuid] = list(entry_kv)
            for e, kv in entry_kv.items():
                self.kv_transfer_server.await_pull(
                    entry_uuid(req_meta.uuid, e), kv)
            if self._early_pull_done.pop(req_meta.uuid, None) is not None:
                self._release_kv(req_id, req_meta.uuid)
        t2 = time.perf_counter()
        self.stats.increment_send(nbytes)
        logger.info("PDSEND " + json.dumps(
            dict(req_id=req_id,
                 uuid=req_meta.uuid,
                 num_tokens=req_meta.num_tokens,
                 entries=len(entry_kv),
                 bytes=nbytes,
                 tail_bytes=sum(k.nbytes
                                for k in entry_kv.get(_TAIL_ENTRY, [])),
                 wire_bytes=nbytes * len(self.xfer_sharding.device_set),
                 gather_ms=round((t1 - t0) * 1e3, 3),
                 register_ms=round((t2 - t1) * 1e3, 3),
                 t=time.time())))

    def _drop_pull_entry(self, uuid: int):
        """Drops every entry P registered for uuid (pulled or not)."""
        for e in self._send_entries.pop(uuid, []):
            try:
                self.kv_transfer_server.await_pull(
                    entry_uuid(uuid, e), [],
                    timeout=datetime.datetime.now())
            except Exception as ex:
                logger.warning(f"TPUConnectorDSV4 drop entry failed | "
                               f"uuid={uuid} entry={e} | {ex}")

    # --------------------------------------------------------------- D side
    def _pull_kv(self, req_id: str, conn: Any, req_meta: LoadMeta):
        assert isinstance(req_meta, LoadMetaDSV4), req_meta
        t0 = time.perf_counter()
        plan = plan_entries(req_meta.num_tokens, req_meta.chunk_tokens,
                            self.groups)
        needed = entries_needed(req_meta.num_tokens, req_meta.chunk_tokens,
                                req_meta.local_hit_tokens, plan)
        pulled: dict[int, list[jax.Array]] = {}
        for e in needed:
            pulled[e] = conn.pull(entry_uuid(req_meta.uuid, e),
                                  self._entry_specs(plan[e]))
        t1 = time.perf_counter()
        timeout_s = dist_utils.get_p2p_wait_pull_timeout()
        timed_out = False
        while not all(a.is_ready() for kv in pulled.values() for a in kv):
            if time.perf_counter() - t1 > timeout_s:
                timed_out = True
                break
            time.sleep(0.0005)
        t2 = time.perf_counter()
        nbytes = sum(a.nbytes for kv in pulled.values() for a in kv)
        self.stats.increment_pull(nbytes)
        info = dict(req_id=req_id,
                    uuid=req_meta.uuid,
                    num_tokens=req_meta.num_tokens,
                    local_hit_tokens=req_meta.local_hit_tokens,
                    entries_total=len(plan),
                    entries_pulled=len(needed),
                    bytes=nbytes,
                    tail_bytes=sum(a.nbytes
                                   for a in pulled.get(_TAIL_ENTRY, [])),
                    wire_bytes=nbytes * len(self.xfer_sharding.device_set),
                    issue_ms=round((t1 - t0) * 1e3, 3),
                    wait_ms=round((t2 - t1) * 1e3, 3),
                    pull_ms=round((t2 - t0) * 1e3, 3),
                    timed_out=timed_out,
                    t=time.time())
        if timed_out:
            logger.error("PDPULL " + json.dumps(info))
        else:
            logger.info("PDPULL " + json.dumps(info))
        return (plan, pulled)

    def _insert(self, req_id: str, uuid: int, plan_pulled, local_block_ids,
                local_hit_tokens: int):
        plan, pulled = plan_pulled
        t0 = time.perf_counter()
        # Per array: rows to write (destination block id, value row source).
        per_array: dict[int, tuple[list[np.ndarray], list[jax.Array]]] = {}
        skipped_hit = skipped_null = written = 0
        for e, kv in pulled.items():
            eplan = plan[e]
            for (u, gid), val in zip(self._entry_units(eplan), kv):
                bs = self.groups[gid].block_size
                local = local_block_ids[gid]
                dest = np.empty(len(eplan[gid]), dtype=np.int32)
                for j, i in enumerate(eplan[gid]):
                    lid = local[i] if i < len(local) else self.null_block_id
                    if lid == self.null_block_id:
                        dest[j] = -1
                        skipped_null += 1
                    elif i * bs < local_hit_tokens:
                        dest[j] = -1
                        skipped_hit += 1
                    else:
                        dest[j] = lid
                        written += 1
                a = self.units[u][0]
                per_array.setdefault(a, ([], []))
                per_array[a][0].append(dest)
                per_array[a][1].append(val)
        kv_caches = list(self.runner.kv_caches)
        for a, (dests, vals) in per_array.items():
            num_blocks = kv_caches[a].shape[0]
            dest = np.concatenate(dests)
            # Out-of-range rows are dropped by the scatter.
            dest = np.where(dest < 0, num_blocks, dest).astype(np.int32)
            rows = len(dest)
            padded = 1 << max(0, (rows - 1).bit_length())
            dest = np.pad(dest, (0, padded - rows), constant_values=num_blocks)
            vals = tuple(
                jax.device_put(v, self.insert_sharding) for v in vals)
            kv_caches[a] = _scatter(kv_caches[a], jnp.asarray(dest), vals,
                                    padded)
        if os.getenv("DSV4_PD_SKIP_INSERT") == "1":
            logger.warning("DSV4_PD_SKIP_INSERT=1: not writing pulled KV")
        else:
            self.runner.kv_caches = kv_caches
        jax.block_until_ready(kv_caches)
        t1 = time.perf_counter()
        if _DEBUG:
            for e, kv in pulled.items():
                eplan = plan[e]
                res = []
                for (u, gid), val in zip(self._entry_units(eplan), kv):
                    local = local_block_ids[gid]
                    keep = [j for j, i in enumerate(eplan[gid])
                            if i < len(local) and local[i] != 0
                            and i * self.groups[gid].block_size >=
                            local_hit_tokens]
                    ids = [local[eplan[gid][j]] for j in keep]
                    back = np.asarray(
                        _gather((self.runner.kv_caches[self.units[u][0]], ),
                                (jnp.asarray(ids, dtype=jnp.int32), ))[0])
                    want = np.asarray(val)[np.asarray(keep, dtype=np.int64)]
                    res.append(dict(unit=(self.units[u][0], gid),
                                    pulled=_checksum(val),
                                    rows=len(keep),
                                    equal=bool(np.array_equal(back, want))))
                logger.info("PDDEBUG insert " + json.dumps(
                    dict(uuid=uuid, entry=e, units=res)))
        logger.info("PDINSERT " + json.dumps(
            dict(req_id=req_id,
                 uuid=uuid,
                 insert_ms=round((t1 - t0) * 1e3, 3),
                 written_blocks=written,
                 skipped_hit_blocks=skipped_hit,
                 skipped_null_blocks=skipped_null,
                 t=time.time())))

    def process_send_load(self, metadata):
        for req_id, req_meta in metadata.reqs_to_send.items():
            assert self.is_producer
            self._prepare_kv_and_wait(req_id, req_meta)

        for req_id, req_meta in metadata.reqs_to_load.items():
            assert not self.is_producer
            if req_meta.remote_block_ids is not None:
                conn = self._maybe_build_kv_connection(req_meta)
                if req_id not in self.reqs_pulling:
                    self.reqs_pulling[req_id] = [
                        self.pull_executor.submit(self._pull_kv, req_id, conn,
                                                  req_meta), None,
                        (req_meta.local_block_ids, req_meta.local_hit_tokens)
                    ]
                else:
                    self.reqs_pulling[req_id][2] = (
                        req_meta.local_block_ids, req_meta.local_hit_tokens)
            else:
                if req_id in self.reqs_pulling:
                    assert self.reqs_pulling[req_id][1] is not None
                    _, plan_pulled, (load_ids,
                                     hit) = self.reqs_pulling.pop(req_id)
                    local_ids = req_meta.local_block_ids or load_ids
                    if local_ids != load_ids:
                        logger.warning(
                            f"TPUConnectorDSV4 Decode --> block ids changed "
                            f"between load and insert | req_id={req_id} | "
                            f"load={load_ids} | insert={local_ids}")
                    self._insert(req_id, req_meta.uuid, plan_pulled, local_ids,
                                 hit)
                else:
                    logger.info(f"TPUConnectorDSV4 Worker {self.node_id} "
                                f"Decode --> skip insert | req_id={req_id}")
                socket = self._maybe_build_notif_socket(req_meta)
                self._notify_pull_done(socket, req_id, req_meta.uuid)

    def _maybe_build_notif_socket(self, req_meta: LoadMeta) -> zmq.Socket:
        port = getattr(req_meta, "remote_side_port",
                       None) or self.side_channel_port
        remote_host = req_meta.remote_host
        if isinstance(remote_host, list):
            remote_host = remote_host[self.node_id]
        sock_path = make_zmq_path("tcp", remote_host, port)
        sock = self.notif_sockets.get(sock_path)
        if sock is None:
            sock = make_zmq_socket(ctx=self.zmq_cxt,
                                   path=sock_path,
                                   socket_type=zmq.DEALER,
                                   bind=False)
            self.notif_sockets[sock_path] = sock
        return sock


_DEBUG = os.getenv("DSV4_PD_DEBUG") == "1"


def _checksum(x):
    """Sum of bytes; per device for arrays with several addressable shards."""
    shards = getattr(x, "addressable_shards", None)
    if shards and len(shards) > 1:
        return [int(np.asarray(sh.data).astype(np.uint64).sum())
                for sh in shards]
    return int(np.asarray(x).astype(np.uint64).sum())


def _replicated(sharding):
    if isinstance(sharding, jax.sharding.NamedSharding):
        return jax.sharding.NamedSharding(sharding.mesh,
                                          jax.sharding.PartitionSpec())
    return sharding


@jax.jit
def _gather(arrays: tuple, indices: tuple) -> list[jax.Array]:
    return [a[i] for a, i in zip(arrays, indices)]


@functools.partial(jax.jit, donate_argnums=(0, ), static_argnums=(3, ))
def _scatter(cache: jax.Array, dest: jax.Array, vals: tuple,
             padded: int) -> jax.Array:
    v = jnp.concatenate(vals, axis=0) if len(vals) > 1 else vals[0]
    if v.shape[0] < padded:
        v = jnp.pad(v, [(0, padded - v.shape[0])] + [(0, 0)] *
                    (v.ndim - 1))
    return cache.at[dest].set(v.astype(cache.dtype), mode="drop")
