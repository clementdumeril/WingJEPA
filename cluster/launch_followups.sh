#!/bin/bash
# Follow-up experiments of the SkyJEPA battery on the advanced_plane (after the adv / ups runs). Run ONCE, from the
# repository root on the login node; it only submits jobs (job ids in runs/followups_jobs.txt):
#   bash cluster/launch_followups.sh                 # every part
#   bash cluster/launch_followups.sh cl real         # some parts only
# Parts, in submission order (the first ones are the most useful):
#   cl    closed loop (MPPI, SkyJEPA's horizon 20 and 512 samples): MPPI settings tuned on flight seed 0 with a0 and
#         jepapi (scripts/pick_mppi.py), then frozen; mlp, a0, a0pi, jepa, jepapi, cdirect (+ TECS) of each seed and
#         dataset fly 200 aircraft for 60 s on flight seed 1, in 4 conditions (clean, Gaussian noise 0.2, colored
#         noise 0.2, moderate turbulence + 2 m/s gusts), noise in the units of runs/mlp_adv_s0
#   real  evaluation in common units (std of runs/mlp_adv_s0) with realistic sensor noise and the bias / colored
#         kinds, every adv / ups model on its own test set and on the other one (CPU)
#   zabl  latent ablation of the physics heads (a0pi, jepapi, cdirect): latent shuffled across rollouts or zeroed (CPU)
#   obs   pilot of the partially observed setting: alpha, beta unmeasured, realistic sensor noise; mlp, a0, jepa and
#         the asymmetric JEPA (privileged target encoder), 3 seeds, both datasets
#   noise noise-augmented training (--input-noise 0.5) of mlp, a0, jepa, a0pi, jepapi on data_advanced
#   frac  data efficiency: 1 % and 10 % of the data_advanced flights, mlp, a0, jepa, a0pi, jepapi, cdirect, 3 seeds
#         (lower priority: --nice)
# Runs with a physics head get a chained retry job that continues them if the 8 h limit stopped them.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/slurm runs/closed_loop/tune
D=/global/scratch/users/$USER/WingJEPA/data
DA=$D/data_advanced
DU=$D/data_advanced_upset
PR=configs/priors/advanced_plane_nul.json
S="--H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 --action-encoder 1 --T 20 --batch 2048 --warmup 4000 --steps1 24000 --sigreg-steps 20 --device cuda"
GW="-A ic_piml --partition=savio3_gpu --qos=a40_gpu3_ica --gres=gpu:A40:1 -c 8 --time=08:00:00 --exclude=n0214.savio3,n0215.savio3,n0262.savio3,n0263.savio3,n0264.savio3"
C="-A ic_piml -p savio4_htc -c 8 --mem=64G -t 04:00:00 --exclude=n0036.savio4,n0064.savio4,n0150.savio4,n0232.savio4"
GPUENV="source cluster/compiler.sh; source /global/scratch/users/$USER/venvs/wingjepa-gpu/bin/activate"
CPUENV="source cluster/compiler.sh; source /global/scratch/users/$USER/venvs/wingjepa/bin/activate"
REF=runs/mlp_adv_s0                       # common unit (training std) of the noise levels and normalized errors
PARTS=" ${*:-cl real zabl obs noise frac} "
LOG=runs/followups_jobs.txt
want() { [[ "$PARTS" == *" $1 "* ]]; }
sub() {                                   # sbatch, the job id logged with its name
    local id
    id=$(sbatch --parsable "$@") || { echo "SUBMISSION FAILED: $*" | tee -a $LOG >&2; return 1; }
    echo "$id"
}
data_of() { [ "$1" = ups ] && echo $DU || echo $DA; }
for r in $REF runs/a0_adv_s0 runs/jepapi_adv_s0; do
    [ -f $r/model.pt ] || { echo "missing $r/model.pt: run this from the WingJEPA directory on Savio"; exit 1; }
done
echo "== $(date) parts:$PARTS" >> $LOG

# run NAME DATA DEPENDENCY NICE "EVAL OPTIONS" TRAIN_A OPTIONS...: train + evaluate on the GPU; physics heads get a
# retry job. EV_EXTRA (optional): a second evaluation of the run. Sets LAST = the job that ends the run.
EV_EXTRA=""
run() {
    local n=$1 data=$2 dep=$3 nice=$4 ev=$5
    shift 5
    local tr="python scripts/train_a.py --data-dir $data --set train --out runs/$n $*"
    local evc="python scripts/eval_skyjepa.py --data-dir $data --models runs/$n $ev"
    [ -n "$EV_EXTRA" ] && evc="$evc && python scripts/eval_skyjepa.py --data-dir $data --models runs/$n $EV_EXTRA"
    local j
    j=$(sub $GW $dep $nice -J $n -o runs/slurm/fu-$n-%j.out --wrap "$GPUENV && $tr && $evc") || return
    LAST=$j
    if [[ "$*" == *"--mode c_"* ]]; then
        LAST=$(sub $GW $nice -J $n-retry --dependency=afterany:$j -o runs/slurm/fu-$n-retry-%j.out \
            --wrap "$GPUENV && if [ -f runs/$n/resume.pt ]; then $tr && $evc; else echo already finished; fi") || return
    fi
    echo "$n $j $LAST" >> $LOG
}

if want cl; then
    cmds=""
    for lam in 0.03 0.1 0.3; do
        for sc in 0.5 1 2; do
            sig=$(awk -v s=$sc 'BEGIN{printf "%g %g %g %g", 0.04*s, 0.04*s, 0.03*s, 0.08*s}')
            cmds="$cmds python scripts/closed_loop.py --config configs/data_advanced.yaml --controllers runs/a0_adv_s0 runs/jepapi_adv_s0 --out runs/closed_loop/tune/lam${lam}_sig${sc}.json --n-train 20 --n-test 20 --duration 30 --seed 0 --samples 512 --horizon 20 --lam $lam --sigma $sig;"
        done
    done
    TUNE=$(sub $GW -J cl-tune -o runs/slurm/cl-tune-%j.out \
        --wrap "$GPUENV && $cmds python scripts/pick_mppi.py runs/closed_loop/tune/*.json --out runs/closed_loop/mppi_best.json")
    echo "cl-tune $TUNE" >> $LOG
    declare -A COND=([clean]="" [gauss02]="--noise 0.2" [colored02]="--noise 0.2 --noise-kind colored"
                     [turb]="--turbulence moderate --gust 2")
    for ds in adv ups; do
        for s in 0 1 2; do
            M=$(for r in mlp a0 a0pi jepa jepapi cdirect; do printf " runs/${r}_${ds}_s$s"; done)
            for c in clean gauss02 colored02 turb; do
                j=$(sub $GW --dependency=afterok:$TUNE -J cl-$ds-s$s-$c -o runs/slurm/cl-$ds-s$s-$c-%j.out \
                    --wrap "$GPUENV && python scripts/closed_loop.py --config configs/data_advanced.yaml --controllers tecs$M --out runs/closed_loop/${ds}_s${s}_${c}.json --n-train 100 --n-test 100 --duration 60 --seed 1 --samples 512 --horizon 20 --params-from runs/closed_loop/mppi_best.json --noise-std-from $REF ${COND[$c]}")
                echo "cl-$ds-s$s-$c $j" >> $LOG
            done
        done
    done
fi

if want real; then
    RE="--realistic-noise 1 --std-from $REF --noise-kinds gauss bias colored"
    for ds in adv ups; do
        M=$(ls -d runs/*_${ds}_s[0-2] | tr '\n' ' ')
        for on in adv ups; do
            tag=_real; [ $on != $ds ] && tag=_real_on_$([ $on = ups ] && echo upset || echo advanced)
            j=$(sub $C -J real-$ds-on-$on -o runs/slurm/real-$ds-on-$on-%j.out \
                --wrap "$CPUENV && python scripts/eval_skyjepa.py --data-dir $(data_of $on) --models $M $RE --tag $tag")
            echo "real-$ds-on-$on $j" >> $LOG
        done
    done
fi

if want zabl; then
    for ds in adv ups; do
        M=$(for r in a0pi jepapi cdirect; do for s in 0 1 2; do printf " runs/${r}_${ds}_s$s"; done; done)
        E="python scripts/eval_skyjepa.py --data-dir $(data_of $ds) --models$M"
        j=$(sub $C -J zabl-$ds -o runs/slurm/zabl-$ds-%j.out \
            --wrap "$CPUENV && $E --z-ablation shuffle --tag _zshuffle && $E --z-ablation zero --tag _zzero")
        echo "zabl-$ds $j" >> $LOG
    done
fi

if want obs; then
    for ds in adv ups; do
        data=$(data_of $ds)
        for s in 0 1 2; do
            O="--seed $s $S --obs realistic"
            run mlp_${ds}obs_s$s $data "" "" "--obs realistic --tag _obs" --mode mlp_direct $O --mlp-hidden 50,50
            run a0_${ds}obs_s$s $data "" "" "--obs realistic --tag _obs" --mode a0 $O
            run jepa_${ds}obs_s$s $data "" "" "--obs realistic --tag _obs" --mode jepa $O
            run jepaasym_${ds}obs_s$s $data "" "" "--obs realistic --tag _obs" --mode jepa $O --target-priv 1
        done
    done
fi

if want noise; then
    EV_EXTRA="--realistic-noise 1 --std-from $REF --noise-kinds gauss bias colored --tag _real"
    for s in 0 1 2; do
        N="--seed $s $S --input-noise 0.5"
        run mlp_advn_s$s $DA "" "" "" --mode mlp_direct $N --mlp-hidden 50,50
        run a0_advn_s$s $DA "" "" "" --mode a0 $N
        A=$LAST
        run jepa_advn_s$s $DA "" "" "" --mode jepa $N
        J=$LAST
        run a0pi_advn_s$s $DA "--dependency=afterok:$A" "" "" --mode c_jepa $N --prior $PR --trunk runs/a0_advn_s$s
        run jepapi_advn_s$s $DA "--dependency=afterok:$J" "" "" --mode c_jepa $N --prior $PR --trunk runs/jepa_advn_s$s
    done
    EV_EXTRA=""
fi

if want frac; then
    for f in 0.01 0.1; do
        t=adv$(awk -v f=$f 'BEGIN{print f * 100}')pc
        for s in 0 1 2; do
            F="--seed $s $S --train-frac $f"
            run mlp_${t}_s$s $DA "" "--nice=100" "" --mode mlp_direct $F --mlp-hidden 50,50
            run a0_${t}_s$s $DA "" "--nice=100" "" --mode a0 $F
            A=$LAST
            run jepa_${t}_s$s $DA "" "--nice=100" "" --mode jepa $F
            J=$LAST
            run a0pi_${t}_s$s $DA "--dependency=afterok:$A" "--nice=100" "" --mode c_jepa $F --prior $PR --trunk runs/a0_${t}_s$s
            run jepapi_${t}_s$s $DA "--dependency=afterok:$J" "--nice=100" "" --mode c_jepa $F --prior $PR --trunk runs/jepa_${t}_s$s
            run cdirect_${t}_s$s $DA "" "--nice=100" "" --mode c_direct $F --prior $PR
        done
    done
fi

echo "submitted: $(grep -c . $LOG) lines in $LOG; queue: $(squeue -u $USER -h | wc -l) jobs"
