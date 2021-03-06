from setuptools import setup, find_packages
import sys


extras = {
    'test': [
        'filelock', 
        'pytest'
    ]
}


all_deps = []
for group_name in extras:
    all_deps += extras[group_name]

extras['all'] = all_deps

setup(name='baselines',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      install_requires=[
          'gym[mujoco,atari,classic_control,robotics]',
          'scipy',
          'tqdm',
          'joblib',
          'dill',
          'progressbar2',
          'mpi4py',
          'cloudpickle',
          'click',
          'opencv-python',
          'backports.tempfile',
          'six'
      ],
      extras_require=extras,
      description='OpenAI baselines: high quality implementations of reinforcement learning algorithms',
      author='OpenAI',
      url='https://github.com/openai/baselines',
      author_email='gym@openai.com',
      version='0.1.5')


# ensure there is some tensorflow build with version above 1.4
try:
    from distutils.version import StrictVersion
    import tensorflow
    assert StrictVersion(tensorflow.__version__) >= StrictVersion('1.4.0')
except ImportError:
    assert False, "TensorFlow needed, of version above 1.4"
