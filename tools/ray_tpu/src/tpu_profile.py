# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow
import socket
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "profile_dir", None, "Directory to store profile data."
)
flags.DEFINE_integer("time_ms", None, "Profile time lenght in ms.")
flags.DEFINE_integer("port", None, "Profile port.")

def main(_):
  hostname = socket.gethostname()
  ip_addr = socket.gethostbyname(hostname)
  tensorflow.profiler.experimental.client.trace(
      f"{ip_addr}:{FLAGS.port}", FLAGS.profile_dir, 
      FLAGS.time_ms)

if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)