from setuptools import setup

setup(name='learn2sample',
      version='0.1',
      description='Implementation of Various adaptive MCMC samplers',
      url='http://github.com/AVMCMC/,
      author='anonymous',
      author_email='none',
      packages=['LearningToSample'],
      install_requires=['numpy', 'tensorflow', 'matplotlib',
                        'tensorflow_probability'],
      zip_safe=False)