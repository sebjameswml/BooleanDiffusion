#ifdef COPYLEFT
/*
 *  This file is part of BooleanDiffusion.
 *
 *  BooleanDiffusion is free software: you can redistribute it and/or modify it under
 *  the terms of the GNU General Public License as published by the Free Software
 *  Foundation, either version 3 of the License, or (at your option) any later version.
 *
 *  BooleanDiffusion is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *  PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with
 *  BooleanDiffusion.  If not, see <https://www.gnu.org/licenses/>.
 */
#endif

/*
 * This file provides the class RD_Bool2.
 *
 * See paper/supp.tex
 *
 * Author: Seb James
 * Date: November 2020
 */

#include "rd_bool.h"
#include <vector>
#include <iostream>

/*!
 * Reaction diffusion system in which the reaction is an NK model (a Boolean gene
 * regulatory network).
 */
template <typename Flt, size_t N, size_t K>
class RD_Bool2 : public RD_Bool<Flt, N, K>
{
public:
    //! Time of state change, in steps. If -1, then state s may change, otherwise,
    //! stepCount must exceed tsc[h] for hex h to change state.
    alignas(alignof(std::vector<int>)) std::vector<int> tsc;

    //! The expressing state of the hex.
    alignas(alignof(std::vector<morph::bn::state_t>))
    std::vector<morph::bn::state_t> s_e;

    //! How long to delay expression, in timesteps. Equal to 1/(alpha * dt)
    int expression_delay;

    RD_Bool2() : RD_Bool<Flt, N, K>() {}

    virtual void allocate()
    {
        RD_Bool<Flt, N, K>::allocate();
        // Note: Setting tsc to -1 at start
        this->tsc.resize (N, -1);
        this->s_e.resize (N, 0);
    }

    virtual void init()
    {
        std::cout << "RD_Bool2::init()\n";
        RD_Bool<Flt, N, K>::init();
        Flt _alpha = Flt{0};
        for (size_t i = 0; i < N; ++i) {
            _alpha += this->alpha[i];
        }
        _alpha /= Flt{N};
        this->expression_delay = (int) (Flt{1}/(_alpha*this->dt));
    }

    virtual void init_a()
    {
        std::cout << "RD_Bool2::init_a()\n";
        this->gauss.gain = 1.0;
        this->gauss.sigma = 0.05;
        this->gauss.sigmasq = this->gauss.sigma * this->gauss.sigma;
        this->gauss.x = 0.05;
        this->gauss.y = 0;

        this->set_vector_vector (this->a, N, this->expression_threshold);

        // Initialise a[N-1] which is 'Gene a'
        for (auto h : this->hg->hexen) {
            Flt dsq = morph::MathAlgo::distance_sq<Flt> ({this->gauss.x, this->gauss.y}, {h.x, h.y});
            this->a[N-1][h.vi] = this->gauss.gain * std::exp (-dsq / (Flt{2} * this->gauss.sigmasq));
        }

#if 1
        // a[n-2] or 'Gene b'
        this->gauss.x = -0.05;
        for (auto h : this->hg->hexen) {
            Flt dsq = morph::MathAlgo::distance_sq<Flt> ({this->gauss.x, this->gauss.y}, {h.x, h.y});
            this->a[N-2][h.vi] = this->gauss.gain * std::exp (-dsq / (Flt{2} * this->gauss.sigmasq));
        }
#endif
    }

    //! Compute inputs for the gene regulatory network, its next developed step (for
    //! each hex) and its outputs, storing these in this->T and this->s.
    virtual void compute_genenet()
    {
        // 1. Compute T(a_i) in each hex. In each hex, the state may be different
        for (unsigned int h=0; h<this->nhex; ++h) {
            if (this->tsc[h] == -1) {
                this->s[h] = 0x0;
                // Check each gene to find out if its concentration is above threshold.
                for (size_t i = 0; i < N; ++i) {
                    // Set s based on a[i][h]
                    this->s[h] |= (this->a[i][h] > this->expression_threshold ? 0x1 : 0x0) << i;
                    // T is a function that returns the amount by which a is above
                    // (or below if negative) the expression threshold.
                    this->T[i][h] = this->a[i][h] - this->expression_threshold;
                }
                // Now have the current state, see what the next state is. grn.develop() is
                // G() in the notes and this line turns s into s':
                this->grn.develop (this->s[h], this->genome);
                if (this->s[h] != this->s_e[h]) {
                    // Developing state is different from the previous expressing state,
                    // so update it, and set timestamp.
                    std::cout << "Updating state for hex " << h << " at timestep " << this->stepCount << std::endl;
                    this->s_e[h] = this->s[h];
                    this->tsc[h] = static_cast<int>(this->stepCount);
                }
            } else {
                // tsc has a stepCount in it.
                if (static_cast<int>(this->stepCount) - this->tsc[h] > this->expression_delay) {
                    tsc[h] = -1;
                }
            }
        }
    }

    static constexpr bool debug_compute_dadt = false;

    virtual void compute_dadt (const size_t i, std::vector<Flt>& a_, std::vector<Flt>& dadt)
    {
        std::vector<Flt> lap_a(this->nhex, 0.0);
        this->compute_laplace (a_, lap_a);
//#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            Flt term1 = this->D[i] * lap_a[h];
            Flt term2 = - this->alpha[i] * a_[h];

            // s[h] is the current state of 'expressingness' for each gene, but should
            // be modulated by the levels of reagents available. G[i][h] contains the
            // input reagent levels of which we choose the minimum one (? or an
            // average?) to modulate how much those genes that are being expressed ARE
            // actually being expressed. For G's that are below threshold the 'G' is the
            // amount by which G is below the threshold.

            Flt _F = Flt{0};
            for (unsigned int j = 0; j<N; ++j) {
                if constexpr (debug_compute_dadt) {
                    if (h == 0) {
                        std::cout << "Adding " << (this->T[j][h] * this->T[j][h])
                                  << " to _F for Gene " << j << std::endl;
                    }
                }
                _F += this->T[j][h] * this->T[j][h];
            }

            // F is RMS of T squared or 0, depending on s_e[h] being 1 or 0
            this->F[i][h] = (this->s_e[h] & 1<<i) ? std::sqrt (_F / Flt{N}) : Flt{0};

            // Term 3 is the output expression for gene i
            if constexpr (debug_compute_dadt) {
                if (h == 0) {
                    std::cout << "F[i][h=0]: " << this->F[i][h];
                    std::cout << ", s_e[" << h << "] = " << morph::bn::GeneNet<N,K>::state_str(this->s_e[h]) << std::endl;
                }
            }

            Flt term3 = this->beta[i] * this->F[i][h];

            dadt[h] = term1 + term2 + term3;
            if constexpr (debug_compute_dadt) {
                if (h == 0) {
                    std::cout << "dadt["<<i<<"]["<<h<<"] = "
                              << term1 << " + " << term2 << " + " << term3 << " = " << dadt[h] << std::endl;
                    std::cout << " (a["<<i<<"]["<<h<<"] = " << this->a[i][h] << ")\n";
                }
            }
        }
    }

}; // RD_Bool2
