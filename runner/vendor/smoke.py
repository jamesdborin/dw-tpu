"""Run simultaneously on each granted host; never run without a TPU lease."""
import argparse, functools, json, os
p=argparse.ArgumentParser()
p.add_argument('--coordinator',required=True)
p.add_argument('--rank',type=int,required=True)
p.add_argument('--hosts',type=int,default=8)
p.add_argument('--cpu-test',action='store_true')
a=p.parse_args()
os.environ['JAX_PLATFORMS']='cpu' if a.cpu_test else 'tpu'
if a.cpu_test: os.environ['XLA_FLAGS']='--xla_force_host_platform_device_count=4'
import jax
import jax.numpy as jnp
import numpy as np
jax.distributed.initialize(coordinator_address=a.coordinator,num_processes=a.hosts,process_id=a.rank,initialization_timeout=180)
assert jax.device_count()==a.hosts*4, jax.devices()
assert jax.local_device_count()==4
@functools.partial(jax.pmap,axis_name='chips')
def collective(x):
 return jax.lax.psum(x,'chips')
x=jnp.arange(a.rank*4,a.rank*4+4,dtype=jnp.int32)
y=np.asarray(collective(x))
expected=sum(range(a.hosts*4))
assert np.all(y==expected),(y,expected)
print(json.dumps({'rank':a.rank,'devices':jax.device_count(),'local_devices':4,'psum':y.tolist(),'expected':expected,'passed':True}),flush=True)
jax.distributed.shutdown()
