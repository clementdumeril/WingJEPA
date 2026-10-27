# Getting started on Savio (BRC)

Official documentation: https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/

> Login takes your PIN plus a Google Authenticator code, so these steps are manual. Nothing in this repository stores
> credentials. Points marked "to check" are not confirmed from the docs.

## 1. First login (once)

1. Set up the OTP token (Google Authenticator); see *Setting Up One-Time Passwords* in the docs.
2. From a terminal:
   ```bash
   ssh <savio_login>@hpc.brc.berkeley.edu
   # Password: the PIN immediately followed by the 6-digit code
   ```
3. Once logged in, find your Slurm account (it starts with `fc_` or `co_`) and the partitions you can use:
   ```bash
   sacctmgr -p show associations user=$USER     # to check: lists accounts / partitions / QoS
   sinfo -s                                     # available partitions
   ```
   The `ACCOUNT` (e.g. `fc_xxx`) goes in the `--account` line of the `.sbatch` scripts.

## 2. Login without a password (optional, 12-hour certificate)

On Windows the docs recommend WSL (Ubuntu). In a WSL terminal:
```bash
git clone https://github.com/lbnl-science-it/lrc-scripts.git
cd lrc-scripts
./request_cert.sh -p brc          # asks for login, PIN, then OTP
```
Then add this to `~/.ssh/config` (on the Windows side, with the path to the certificate inside WSL):
```
Host savio
    User <savio_login>
    HostName hpc.brc.berkeley.edu
    IdentityFile //wsl$/Ubuntu/home/<wsl_user>/.ssh/ssh_certs/brc_cert
    IdentitiesOnly yes
```
Then `ssh savio` works until the certificate expires (12 hours).

## 3. Install the environment on Savio (once)

```bash
ssh savio
cd /global/scratch/users/$USER            # to check: scratch space (large, not backed up)
git clone https://github.com/clementdumeril/WingJEPA.git
cd WingJEPA
bash cluster/setup_env.sh                  # creates the Python environment in scratch
```

## 4. Running jobs

Never run computations on the login node; always go through Slurm.

```bash
sbatch cluster/job_cpu.sbatch scripts/phase0_jsbsim_checks.py     # a test CPU job
sbatch --array=0-134 cluster/job_array.sbatch configs/sweep.yaml  # the experiment grid (phase 3)
squeue -u $USER                                                    # check your jobs
```

The workload (JSBSim, models of ~100 k parameters) is mostly CPU, so the HTC partitions (`savio3_htc`, `savio4_htc`)
fit.

A training job stopped by the time limit can be resubmitted with the same command: it continues from
`runs/<run>/resume.pt`, saved every 15 minutes (`cluster/smoke_resume.sbatch` checks that a resumed run matches a run
done in one go).

To compare CPU and GPU (JEPA spends most of its time encoding the target windows, a large batched convolution):
```bash
bash cluster/setup_gpu_env.sh            # once, on the login node: environment with a CUDA build of torch
sbatch cluster/bench_device.sbatch       # same outputs on GPU and CPU, then seconds per step for each model
```

## 5. Getting results back

Use the transfer node `dtn.brc.berkeley.edu`, not the login node:
```bash
scp -r <savio_login>@dtn.brc.berkeley.edu:/global/scratch/users/<savio_login>/WingJEPA/runs ./runs
```

## My configuration (checked Oct 15, 2026)

- Login `<savio_login>`, Slurm account `<slurm_account>` (the course's instructional computing allocation).
- CPU: `savio3_htc`, `savio4_htc` (QoS `savio_normal`, `savio_debug`). The scripts default to `savio4_htc`.
- GPU if needed: `savio3_gpu` (QoS `a40_gpu3_ica`, `v100_gpu3_ica`, `gtx2080_gpu3_ica`),
  `savio4_gpu` (QoS `a5k_gpu4_ica`).
- Working directory: `/global/scratch/users/<savio_login>/WingJEPA`.
