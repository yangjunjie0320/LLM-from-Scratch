env=llm-from-scratch
if ! conda env list | grep -q "$env"; then
    conda env create -f environment.yml -n $env
fi

conda activate $env

export MKL_NUM_THREADS=20
export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
