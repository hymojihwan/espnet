# ====== About run.pl, queue.pl, slurm.pl, and ssh.pl ======
# Usage: <cmd>.pl [options] JOB=1:<nj> <log> <command...>
# e.g.
#   run.pl --mem 4G JOB=1:10 echo.JOB.log echo JOB
#
# Options:
#   --time <time>: Limit the maximum time to execute.
#   --mem <mem>: Limit the maximum memory usage.
#   -–max-jobs-run <njob>: Limit the number parallel jobs. This is ignored for non-array jobs.
#   --num-threads <ngpu>: Specify the number of CPU core.
#   --gpu <ngpu>: Specify the number of GPU devices.
#   --config: Change the configuration file from default.

cmd_backend='local'

# "qsub.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

# Run locally:
export train_cmd="run.pl"
export cuda_cmd="run.pl"
export decode_cmd="run.pl"

