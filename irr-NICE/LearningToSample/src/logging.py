import os
import errno

def make_run_dir(log_dir):
    """Generates a new numbered directory for this run to store logs"""
    try:
        os.makedirs(log_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    run_num = (sum(os.path.isdir(os.path.join(log_dir,i))
                  for i in os.listdir(log_dir)) + 1)
    run_dir = os.path.join(log_dir, f'run{run_num}')
    if os.path.isdir(run_dir):
        raise OSError(f'Directory: {run_dir} already exists, exiting!')
    else:
        print(f'Creating directory for new run: {run_dir}')
        os.makedirs(run_dir)
        os.makedirs(os.path.join(run_dir, 'run_info'))
        os.makedirs(os.path.join(run_dir, 'run_ckpts'))
        os.makedirs(os.path.join(run_dir, 'run_logs'))
        os.makedirs(os.path.join(run_dir, 'run_results'))
        os.makedirs(os.path.join(run_dir, 'run_figs'))

    return {'info': os.path.join(run_dir, 'run_info'),
            'ckpts': os.path.join(run_dir, 'run_ckpts'),
            'results': os.path.join(run_dir, 'run_results'),
            'figs': os.path.join(run_dir, 'run_figs'),
            'logs': os.path.join(run_dir, 'run_logs')
            }

