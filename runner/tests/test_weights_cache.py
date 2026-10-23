import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import weights_cache as cache
from load_timing import analyze


def identity(shards=None):
    return {'checkpoint_sha256':'a'*64, 'processing_sha256':'b'*64,
            'quantization':{'expert':'fp4','block_size':32}, 'layout_version':1,
            'runtime_versions':{'jax':'0.11.2'}, 'topology':{'mesh':[4,8], 'axis_order':['tp']},
            'host_index':0, 'host_count':8, 'model_role':'target', 'shard_schema_sha256':cache.schema_digest(shards or [metadata()])}


def metadata():
    return {'tensor':'layer.weight', 'dtype':'float32', 'global_shape':[8], 'local_shape':[2],
            'index':[[0,2,1]], 'device_slot':0, 'layout':{'major_to_minor':[0]}, 'storage_bytes':8}


class WeightsCacheTests(unittest.TestCase):
    def test_round_trip_streams_postprocessed_local_shards(self):
        data=struct.pack('<ff', 1.25, -3.5)
        with tempfile.TemporaryDirectory() as tmp:
            path=cache.publish(tmp, identity(), [(metadata(), cache.local_array_chunks(data))])
            restored=[]
            result=cache.restore(tmp, identity(), lambda meta, buffer: restored.append(struct.unpack('<ff', buffer)))
            self.assertEqual(restored, [(1.25,-3.5)])
            self.assertEqual(result['shards'][0]['dtype'],'float32')
            self.assertTrue((path/'COMPLETE').exists())
            self.assertEqual(cache.publish(tmp, identity(), iter(())), path)

    def test_keys_include_processing_checkpoint_topology_runtime_host_and_role(self):
        original=identity(); key=cache.cache_key(original)
        for field,value in [('checkpoint_sha256','c'*64),('processing_sha256','d'*64),('host_index',1),
                            ('quantization',{'expert':'bf16'}),('topology',{'mesh':[2,8]}),
                            ('layout_version',2),('runtime_versions',{'jax':'other'}),('model_role','draft')]:
            changed=dict(original); changed[field]=value
            self.assertNotEqual(cache.cache_key(changed), key)
        with self.assertRaises(ValueError): cache.cache_key({'host_index':0})

    def test_incomplete_write_never_becomes_visible(self):
        def broken():
            yield b'123'
            raise RuntimeError('injected interrupted write')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError): cache.publish(tmp, identity(), [(metadata(),broken())])
            self.assertFalse((Path(tmp)/cache.cache_key(identity())).exists())
            self.assertFalse(any(p.is_dir() for p in Path(tmp).iterdir()))

    def test_verify_all_shards_before_consumer_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            ident=identity([metadata(),dict(metadata(),device_slot=1)])
            path=cache.publish(tmp, ident, [(metadata(),[b'goodgood']), (dict(metadata(),device_slot=1),[b'goodgood'])])
            (path/'000001.bin').write_bytes(b'evilevil')
            consumed=[]
            with self.assertRaisesRegex(ValueError,'corrupt cache shard'):
                cache.restore(tmp,ident,lambda *args:consumed.append(True))
            self.assertEqual(consumed,[])
            (path/'COMPLETE').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'manifest'): cache.verified_manifest(tmp,ident)

    def test_duplicate_shards_and_unbounded_chunks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'duplicate'):
                cache.publish(tmp,identity(),[(metadata(),[b'12345678']),(metadata(),[b'12345678'])])
            with self.assertRaisesRegex(ValueError,'bounded'):
                cache.publish(tmp,identity(),[(metadata(),[bytes(cache.CHUNK_BYTES+1)])])

    def test_log_spans_do_not_invent_fp4_or_transfer_split(self):
        lines=['(Worker pid=1) Initializing model took 10.0 seconds',
               '(Worker pid=1) Loading weights took 30.0 seconds',
               '(Worker pid=1) Total time to load model weights from storage to TPU: 600.0 seconds.',
               '(Worker pid=1) Initializing model took 0.2 seconds',
               '(Worker pid=1) Loading weights took 2.0 seconds',
               '(Worker pid=1) Total time to load model weights from storage to TPU: 17.0 seconds.']
        report=analyze(lines)
        self.assertEqual(report['worker_loads'][0]['unseparated_postload_seconds'],560)
        self.assertEqual(report['worker_loads'][1]['index'],1)
        self.assertNotIn('fp4_seconds',report['worker_loads'][0])

    def test_partial_tensor_set_never_publishes_complete(self):
        expected=identity([metadata(),dict(metadata(),device_slot=1)])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'expected shard schema'):
                cache.publish(tmp,expected,[(metadata(),[b'12345678'])])
            self.assertFalse((Path(tmp)/cache.cache_key(expected)).exists())

    def test_phase_timer_synchronizes_both_boundaries(self):
        from load_timing import timed_phase
        events=[]; barriers=[]; ticks=iter([10,13])
        with timed_phase('fp4_process',events.append,lambda:barriers.append(True),lambda:next(ticks)):
            pass
        self.assertEqual(barriers,[True,True])
        self.assertEqual(events[0]['seconds'],3)
