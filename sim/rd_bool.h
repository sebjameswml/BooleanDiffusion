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
 * This file provides the class RD_Bool
 *
 * See paper/supp.tex
 *
 * Author: Seb James
 * Date: November 2020
 */

#include <morph/RD_Base.h>
#include <morph/Hex.h>
#include <morph/HdfData.h>
#include <morph/Random.h>
#include <morph/bn/GeneNet.h>
#include <morph/bn/Genome.h>
#include <morph/MathAlgo.h>

#include <vector>
#include <array>
#include <list>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <cmath>
#include <iostream>

/*!
 * A small collection of parameters to define width and location of a symmetric
 * (i.e. circular) 2D Gaussian.
 */
template <class Flt>
struct GaussParams
{
    Flt gain;
    Flt sigma;
    Flt sigmasq;
    Flt x;
    Flt y;
};

/*!
 * Reaction diffusion system in which the reaction is an NK model (a Boolean gene
 * regulatory network).
 */
template <typename Flt, size_t N, size_t K>
class RD_Bool : public morph::RD_Base<Flt>
{
public:
    //! Our gene regulatory network
    morph::bn::GeneNet<N,K> grn;

    //! A genome for running the simulation
    morph::bn::Genome<N,K> genome;

    //! These are the a_i(x,t) variables/
    alignas(alignof(std::vector<std::vector<Flt> >))
    std::vector<std::vector<Flt> > a;

    //! gradient of a
    alignas(alignof(std::vector<std::array<std::vector<Flt>, 2> >))
    std::vector<std::array<std::vector<Flt>, 2> > grad_a;

    //! alpha_i parameters (decay rates)
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> alpha;

    //! D_i parameters (diffusion constants)
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> D;

    //! Delta_i parameters (accretion rates, when gene is in expressing state)
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> Delta;

    //! To hold a value to say if Gene[i] is above the input expression threshold or
    //! not. Useful for graphing.
    alignas(alignof(std::vector<std::vector<Flt> >))
    std::vector<std::vector<Flt> > G;

    //! Holds the output expression value. For graphing.
    alignas(alignof(std::vector<std::vector<Flt> >))
    std::vector<std::vector<Flt> > H;

    //! The state of the gene network at each hex, computed by calling develop() on the
    //! state constructed by looking at the expression levels of each gene, a[i]
    alignas(alignof(std::vector<morph::bn::state_t>))
    std::vector<morph::bn::state_t> s;

    //! The threshold for it to be considered that a gene is being expressed and is
    //! present.
    Flt expression_threshold = 0.5f;

    //! Shape for init a
    GaussParams<Flt> gauss;

    //! Default constructor
    RD_Bool() : morph::RD_Base<Flt>() {}

    //! Perform memory allocations, vector resizes and so on.
    void allocate()
    {
        // Always call allocate() from the base class first.
        morph::RD_Base<Flt>::allocate();
        // Resize and zero-initialise the various containers. Note that the size of a
        // 'vector variable' is given by the number of hexes in the hex grid which is
        // a member of this class (via its parent, RD_Base)
        this->resize_vector_vector (this->a, N);
        this->resize_vector_vector (this->G, N);
        this->resize_vector_vector (this->H, N);
        this->resize_vector_array_vector (this->grad_a, N);
        this->resize_vector_param (this->alpha, N);
        this->resize_vector_param (this->D, N);
        this->resize_vector_param (this->Delta, N);
        this->s.resize (this->nhex, 0);
    }

    //! Initilization and any one-time computations required of the model.
    void init()
    {
        this->zero_vector_vector (this->G, N);
        this->zero_vector_vector (this->H, N);
        this->zero_vector_array_vector (this->grad_a, N);

        this->zero_vector_vector (this->a, N);
        this->init_a();

        this->genome.randomize();
    }


    void init_a()
    {
        this->gauss.gain = 1.0;
        this->gauss.sigma = 0.05;
        this->gauss.sigmasq = this->gauss.sigma * this->gauss.sigma;
        this->gauss.x = 0.05;
        this->gauss.y = 0;

        // Only initializing a[0] here, which is the *last letter-named gene".
        for (auto h : this->hg->hexen) {
            Flt dsq = morph::MathAlgo::distance_sq<Flt> ({this->gauss.x, this->gauss.y}, {h.x, h.y});
            this->a[0][h.vi] = this->gauss.gain * std::exp (-dsq / (Flt{2} * this->gauss.sigmasq));
            //std::cout << "a[0]["<<h.vi<<"] = " << this->a[0][h.vi] << std::endl;
        }

        // Only initializing a[1] here:
        this->gauss.x = -0.05;
        for (auto h : this->hg->hexen) {
            Flt dsq = morph::MathAlgo::distance_sq<Flt> ({this->gauss.x, this->gauss.y}, {h.x, h.y});
            this->a[1][h.vi] = this->gauss.gain * std::exp (-dsq / (Flt{2} * this->gauss.sigmasq));
        }
    }

    //! Analyse the basins of attraction, making sets of the states in each basin, so
    //! that hexes can be coloured by basin of attraction membership
    void analyse_basins()
    {
    }

    void save()
    {
        std::stringstream fname;
        fname << this->logpath << "/dat_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        morph::HdfData data(fname.str());
        for (unsigned int i = 0; i<N; ++i) {
            std::stringstream path;
            path << "/a_" << i;
            data.add_contained_vals (path.str().c_str(), this->a[i]);
        }
    }

    //! Compute the sum of variable a[_i] (FIXME: Make 'a' a vVector
    Flt sum_a (size_t _i)
    {
        Flt sum = Flt{0};
        for (auto _a : this->a[_i]) { sum += _a; }
        return sum;
    }

    static constexpr bool debug_compute = true;

    Flt k = Flt{1.0};
    Flt sigmoid (Flt _a) { return (Flt{1} / (Flt{1} + std::exp(-this->k*_a))); }

    //! Compute inputs for the gene regulatory network, its next developed step (for
    //! each hex) and its outputs, storing these in this->G
    void compute_genenet()
    {
        // 1. Compute sigma(a_i) in each hex. In each hex, the state may be different
        for (unsigned int h=0; h<this->nhex; ++h) {
            this->s[h] = 0x0;
            // Check each gene to find out if its concentration is above threshold.
            for (size_t i = 0; i < N; ++i) {
                if (this->a[i][h] > this->expression_threshold) {
                    // Then this gene contributes to state. Update this->s.
                    s[h] |= 0x1 << i;
                    // G holds values of a that are above threshold. Used later to
                    // modulate gene production.
                    this->G[i][h] = this->a[i][h]; // This is the input to the GRN
                } else {
                    // This gene does not contribute to state. Non-expressing genes may
                    // need a 'gene production value' too, so that we have a value by
                    // which to module the amount of gene product i that should be
                    // generated in each time step.
                    this->G[i][h] = this->a[i][h] - this->expression_threshold;
                }
            }
            // Now have the current state, see what the next state is
            this->grn.develop (this->s[h], this->genome);
        }
    }

    void compute_dadt (const size_t i, std::vector<Flt>& a_, std::vector<Flt>& dadt)
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
            Flt minpos_G = 1e10;
            Flt minneg_G = -1e10;
            for (unsigned int j = 0; j<N; ++j) {
                minpos_G = (this->G[j][h] > Flt{0} && this->G[j][h] < minpos_G) ? this->G[j][h] : minpos_G;
                minneg_G = (this->G[j][h] <= Flt{0} && this->G[j][h] > minneg_G) ? this->G[j][h] : minneg_G;
            }
            // If no gene was expressing above threshold, min_G needs to be set to -minneg_G
            Flt min_G = (minpos_G == 1e10) ? -minneg_G : minpos_G;

            // Term 3 is the output expression for gene i
#if 0
            if (h == 0) {
                std::cout << "min_G: " << min_G;
                std::cout << ", s[" << h << "] = " << morph::bn::GeneNet<N,K>::state_str(this->s[h]) << std::endl;
            }
#endif
            Flt term3 = (this->s[h] & 1<<i) ? (this->Delta[i] * min_G * min_G) : Flt{0};
            this->H[i][h] = term3;

            dadt[h] = term1 + term2 + term3;
#if 0
            if (h == 0) {
                std::cout << "dadt["<<i<<"]["<<h<<"] = " << term1 << " + " << term2 << " + " << term3 << " = " << dadt[h] << std::endl;
                std::cout << " (a["<<i<<"]["<<h<<"] = " << this->a[i][h] << ")\n";
            }
#endif
        }
    }

    //! Perform one step in the simulation
    void step()
    {
        this->stepCount++;

        this->compute_genenet();

        for (unsigned int i=0; i<N; ++i) {

            // 1. 4th order Runge-Kutta computation for a[i]
            {
                // atst: "a at a test point". Atst is a temporary estimate for A.
                std::vector<Flt> atst(this->nhex, 0.0);
                std::vector<Flt> dadt(this->nhex, 0.0);
                std::vector<Flt> K1(this->nhex, 0.0);
                std::vector<Flt> K2(this->nhex, 0.0);
                std::vector<Flt> K3(this->nhex, 0.0);
                std::vector<Flt> K4(this->nhex, 0.0);

                // Stage 1
                this->compute_dadt (i, this->a[i], dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K1[h] = dadt[h] * this->dt;
                    atst[h] = this->a[i][h] + K1[h] * 0.5 ;
                }

                // Stage 2
                this->compute_dadt (i, atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K2[h] = dadt[h] * this->dt;
                    atst[h] = this->a[i][h] + K2[h] * 0.5;
                }

                // Stage 3
                this->compute_dadt (i, atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K3[h] = dadt[h] * this->dt;
                    atst[h] = this->a[i][h] + K3[h];
                }

                // Stage 4
                this->compute_dadt (i, atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K4[h] = dadt[h] * this->dt;
                }

                // Final sum together. This could be incorporated in the for loop for
                // Stage 4, but I've separated it out for pedagogy.
//#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    Flt delta_a = ((K1[h] + 2.0 * (K2[h] + K3[h]) + K4[h])/(Flt)6.0);
                    this->a[i][h] += delta_a;
                }
            }
        }
    }

}; // RD_Bool
