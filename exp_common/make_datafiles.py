import sys
import os
import hashlib
import struct
import subprocess
import collections
import make_datafiles_cnndm as cnndm

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='cnndm', choices=['cnndm'])
  args, unknown = parser.parse_known_args()

  processors = {'cnndm': cnndm}
  processors[args.dataset].main()
