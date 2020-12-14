#!/bin/zsh

# Explore the 2 gene model, initialised with a Gaussian hump in Gene a.

# Check we're running from BooleanDiffusion dir
CURDIR=$(pwd | awk -F '/' '{print $NF}')

if [ ! $CURDIR = "BooleanDiffusion" ]; then
    echo "Run from the base BooleanDiffusion directory; i.e. ./scripts/$0"
    exit 1
fi

mkdir -p configs/n2
mkdir -p logs/lastframes

# This list of gradient genomes excludes all the degenerate and self-degenerate cases.
for GRADGENOME in 1-0 2-0 0-4 1-4 2-4 0-8 1-8 2-8; do
    for ((gsect1 = 0; gsect1 < 16; gsect1++)); do
        for ((gsect2 = 0; gsect2 < 16; gsect2++)); do

            g1=$(([##16]gsect1))
            g2=$(([##16]gsect2))
            GENOME="${g1}-${g2}"
            JSON=${GENOME}--${GRADGENOME}.json

            cat > configs/n2/${JSON} <<EOF
{
    "steps" : 2000,
    "logevery": 20,
    "plotevery": 20,
    "vidframes": true,
    "finish_asap": true,
    "saveplots": true,
    "overwrite_logs": true,
    "hextohex_d" : 0.02,
    "boundaryFalloffDist" : 0.04,
    "dt" : 0.0005,
    "x_default" : 0.1924,
    "y_default" : -0.001833,
    "z_default" : -0.5,
    "win_width" : 2048,
    "win_height" : 768,
    "map3d" : true,
    "autoscalecolour" : false,

    "graph_mean_ymax" : 0.8,
    "graph_single_ymax" : 1,
    "graph_hexri" : 7,
    "graph_hexgi" : 0,

    "genome" : "${GENOME}",
    "grad_genome" : "${GRADGENOME}",

    "ellipse_a" : 0.15,
    "ellipse_b" : 0.07,

    "expression_threshold" : 0.1,
    "expression_delay" : 80,
    "model_params" : [
        { "alpha" : 0.25, "D" : 0.05, "beta" : 20, "gamma" : 0.2, "name" : "a", "tag" : "MSB" },
        { "alpha" : 0.25, "D" : 0.0005,  "beta" : 20, "gamma" : 0.2, "name" : "b" }
    ],

    "init_a" : [
        {"name" : "a", "idx" : 1, "gain" : 1, "sigma" : 0.05, "x" : 0.05,  "y" : 0, "bg" : 0.05 },
        {"name" : "b", "idx" : 0, "gain" : 0, "sigma" : 0.05, "x" : -0.05, "y" : 0, "bg" : 0.1  }
    ]
}
EOF
            # A version of the sim prog:
            echo "./build/sim/bd2_2 configs/n2/${JSON}"
            ./build/sim/bd2_2 configs/n2/${JSON}
            RTN=$?
            if [ $RTN -ne "0" ]; then
                echo "Config: configs/n2/${JSON} FAILED. Moving on to next."
            fi
            # Copy frames into output folder
            cp logs/${GENOME}--${GRADGENOME}/booldiffusion_0000098.png logs/lastframes/${GENOME}--${GRADGENOME}_98.png
            cp logs/${GENOME}--${GRADGENOME}/booldiffusion_0000099.png logs/lastframes/${GENOME}--${GRADGENOME}_99.png
        done
    done
done

# Success/completion
exit 0
