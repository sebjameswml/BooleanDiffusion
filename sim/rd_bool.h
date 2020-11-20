#ifdef COPYLEFT
/*
 *  This file is part of BooleanDiffusion.
 *
 *  BarrelEmerge is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  BarrelEmerge is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with BarrelEmerge.  If not, see <https://www.gnu.org/licenses/>.
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

    //! Delta_i parameters (accrection rates, when gene is in expressing state)
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> Delta;

    //! To hold the output of the gene network multiplied by Delta. This is a per-hex result
    alignas(alignof(std::vector<std::vector<Flt> >))
    std::vector<std::vector<Flt> > G;

    //! Default constructor
    RD_Bool() : morph::RD_Base<Flt>() {}

    //! Perform memory allocations, vector resizes and so on.
    void allocate() {
        // Always call allocate() from the base class first.
        morph::RD_Base<Flt>::allocate();
        // Resize and zero-initialise the various containers. Note that the size of a
        // 'vector variable' is given by the number of hexes in the hex grid which is
        // a member of this class (via its parent, RD_Base)
        this->resize_vector_vector (this->a, N);
        this->resize_vector_vector (this->G, N);
        this->resize_vector_array_vector (this->grad_a, N);
        this->resize_vector_param (this->alpha, N);
        this->resize_vector_param (this->D, N);
        this->resize_vector_param (this->Delta, N);
    }

    //! Initilization and any one-time computations required of the model.
    void init()
    {
        this->zero_vector_vector (this->a, N);
        this->zero_vector_vector (this->G, N);
        this->zero_vector_array_vector (this->grad_a, N);
        //this->noiseify_vector_vector (this->a, this->initmasks);
        this->a[0][0] = 1.0f;

        this->genome.randomize();
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

    Flt sigmoid (Flt _a) { return (Flt{1} / (Flt{1} + std::exp(-_a))); }

    //! Compute inputs for the gene regulatory network, its next developed step (for
    //! each hex) and its outputs, storing these in this->G
    void compute_genenet()
    {
        // 1. Compute sigma(a_i) in each hex. In each hex, the state may be different
        for (unsigned int h=0; h<this->nhex; ++h) {
            morph::bn::state_t s = 0x0;
            // Check each gene to find out if its concentration is above threshold.
            for (size_t i = 0; i < N; ++i) {
                if (this->sigmoid(a[i][h]) > Flt{0.5}) { s |= 0x1 << i; }
            }
            std::cout << "Start state for hex " << h << " is " << morph::bn::GeneNet<N,K>::state_str(s) << std::endl;
            // Now have the current state, see what the next state is
            this->grn.develop (s, this->genome);
            std::cout << "Next state for hex  " << h << " is             " << morph::bn::GeneNet<N,K>::state_str(s) << std::endl;
            //std::cout << "That means G[][h=" << h << "] = ";
            // Now state contains the 'next state'
            for (size_t i = 0; i < N; ++i) {
                this->G[i][h] = ((s & 1<<i) ? Flt{1} : Flt{0});
                //std::cout << this->G[i][h] << ",";
            }
            //std::cout << std::endl;
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
            Flt term3 = this->Delta[i] * this->G[i][h]
            std::cout << "diffn term: " << term1 << ", decay term: " << term2 << " grn term: " << term3 << std::endl;
            dadt[h] = term1 + term2 + term3;
        }
    }

    //! Perform one step in the simulation
    void step()
    {
        this->stepCount++;

        // Compute the genenet in each hex to find out what this->G should contain.
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
                    std::cout << "For hex " << h << ", added " << delta_a << " to get " << a[i][h] << std::endl;
                }
            }
        }
    }

}; // RD_Bool
