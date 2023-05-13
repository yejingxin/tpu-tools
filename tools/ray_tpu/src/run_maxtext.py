import getpass
import os
import socket
import time
import datetime
from absl import app
from absl import flags
from absl import logging


from ray.job_submission import JobStatus
from ray_tpu_controller import _DEFAULT_RAY_PORT
from ray_tpu_controller import BASE_JAX_PIP_INSTALLS
from ray_tpu_controller import RayTpuController
from ray_tpu_controller import TpuRayJob
from tpu_api import get_default_gcp_project

FLAGS = flags.FLAGS

flags.DEFINE_string('network', 'default', 'VPC network name.')
flags.DEFINE_string('subnet', 'default', 'Subnet name under VPC.')
flags.DEFINE_string(
    'tpu_name',
    getpass.getuser() + '-slice',
    'TPU vm name, it will be suffixed with slice index, like ubuntu-slice-0.',
)
flags.DEFINE_string('command', None,
                    'Main command to run on each TPU.')
flags.DEFINE_string('run_name',  None,
                    'run name to create log dir.')
flags.DEFINE_string('code_dir',  None,
                    'code directory in cpu vm.')
flags.DEFINE_string('tpu_topology', '2x2x2', 'TPU topology.')
flags.DEFINE_integer('num_slices', 2, 'Number of slices.')
flags.DEFINE_enum('mode', None, ['start', 'stop', 'profile'], 'Start or stop running jobs.')
flags.DEFINE_boolean('delete_tpu', False, 'Whether delete tpus when stop jobs.')

"""
python3 run_maxtext.py --code_dir=. --mode=profile \
--command="python3 tpu_profile.py --port=9999 --time_ms=10000 --profile_dir='gs://yejingxin-us-central2/maxtext/profile_0513'"

"""
def stop(controllers):

  for controller in controllers:
    controller.clean_stale_jobs(controller.resource_name)
  logging.info('All Jobs are requested to stop.')
  if FLAGS.delete_tpu:
    for controller in controllers:
      controller.delete_tpu()

def wait_for_jobs_running(controllers):
  num_slices = len(controllers)
  num_slices_job_running = 0
  while num_slices_job_running < num_slices:
    for controller in controllers:
      if controller.jobs_in_status(JobStatus.RUNNING):
        num_slices_job_running += 1
    logging.info(
        '%d/%d slices jobs are running.', num_slices_job_running, num_slices
    )
    time.sleep(10)

  logging.info('All Jobs are in running status, here is the list:')
  for slice_index, controller in enumerate(controllers):
    for job in controller.queued_jobs:
      logging.info('job:%s @ slice: %d', job, slice_index)

def start(controllers):

  for controller in controllers:
    controller.maybe_create_and_wait_for_ready()
    controller.clean_stale_jobs(controller.resource_name)

  mxla_coordinator_address = f'{controllers[0].get_ip_addresses()[0]}:8080'
  num_slices = len(controllers)
  if FLAGS.run_name is None:
    run_name_strings = getpass.getuser() + "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  else:
    run_name_Strings = FLAGS.run_name
  if FLAGS.command is not None:
    run_command = FLAGS.command
  else:
    run_command = f'python3 train.py configs/base.yml run_name={run_name_strings} dcn_data_parallelism={num_slices} ici_data_parallelism=2 enable_checkpointing=True'
  working_dir = os.path.expanduser('~/maxtext/MaxText') #os.path.join(os.getcwd(), 'maxtext')

  pip_installs = {"packages":[
    'jax[tpu]',
    'orbax-checkpoint',
    'absl-py',
    'argparse',
    'datetime',
    'google-cloud-storage',
    'ml-collections',
    'numpy',
    'optax',
    'portpicker',
    'protobuf==3.20.3',
    'pylint',
    'pytest',
    'sentencepiece==0.1.97',
    'tensorflow',
    'tensorflow-datasets',
    'tensorboard-plugin-profile',
    'tensorflow-text',
    'tensorboardx',
    'flax',
    '-f https://storage.googleapis.com/jax-releases/libtpu_releases.html',
  ], 
  "pip_check": False,
      "pip_version": "==20.0.2;python_full_version=='3.8.10'",}


  for slice_index in range(num_slices):
    job = TpuRayJob(
        entrypoint=run_command,
        working_dir=working_dir,
        pip_installs=pip_installs,
        env_vars={
            'LIBTPU_INIT_ARGS': '"--xla_tpu_enable_megascale_barrier=true"',
            'JAX_USE_PJRT_C_API_ON_TPU': '1',
            'MEGASCALE_COORDINATOR_ADDRESS': mxla_coordinator_address,
            'MEGASCALE_SLICE_ID': f'{slice_index}',
            'MEGASCALE_NUM_SLICES': f'{num_slices}',
        },
    )
    controllers[slice_index].queue_tpu_workload(job)
  
  wait_for_jobs_running(controllers)


def profile(controllers):
  for controller in controllers:
    controller.maybe_create_and_wait_for_ready()
  
  num_slices = len(controllers)
  pip_installs = {"packages":[
    'tensorflow',
    'absl-py',
  ], 
  "pip_check": False,
      "pip_version": "==20.0.2;python_full_version=='3.8.10'",}
  for slice_index in range(num_slices):
    job = TpuRayJob(
        entrypoint=FLAGS.command,
        working_dir=FLAGS.code_dir,
        pip_installs=pip_installs,
    )
    controllers[slice_index].queue_tpu_workload(job, resource_name='profile_cpu')
  

def main(_):
  project = get_default_gcp_project()

  hostname = socket.gethostname()
  unused_1, unused_2, unused_3, unused_4, (controller_ip, unused_5) = (
      socket.getaddrinfo(hostname, _DEFAULT_RAY_PORT)[0]
  )

  num_slices = FLAGS.num_slices
  controllers = []
  for slice_index in range(num_slices):
    controller = RayTpuController(
        tpu_name=f'{FLAGS.tpu_name}-{slice_index}',
        project=project,
        zone='us-central2-b',
        accelerator_type='V4',
        accelerator_topology=FLAGS.tpu_topology,
        version='tpu-vm-v4-base',
        network=FLAGS.network,
        subnetwork=FLAGS.subnet,
        head_addr=f'{controller_ip}:{_DEFAULT_RAY_PORT}',
    )
    controllers.append(controller)

  if FLAGS.mode == 'start':
    start(controllers)
  elif FLAGS.mode == 'stop':
    stop(controllers)
  elif FLAGS.mode == 'profile':
    profile(controllers)

if __name__ == '__main__':
  flags.mark_flag_as_required('mode')
  logging.set_verbosity(logging.INFO)
  app.run(main)
