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
 * paper/supplementary/supp.tex; label eq:J_NM_with_comp)
 *
 * This file provides the class RD_Bool
 *
 * Author: Seb James
 * Date: November 2020
 */

#include <morph/RD_Base.h>
//#include <morph/BezCurvePath.h>
#include <morph/Hex.h>
#include <morph/HdfData.h>
#include <morph/Random.h>
#include <morph/bn/GeneNet.h>

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
 * (i.e. circular) 2D Gaussian to set initial values of actors
 */
template <class Flt>
struct GaussParams
{
    Flt gain;
    Flt sigma;
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

    //! J(x,t) - the "flux current". This is a vector field. May need J_A and J_B.
    alignas(alignof(std::array<std::vector<Flt>, 2>))
    std::array<std::vector<Flt>, 2> J;

    //! Default constructor
    RD_Schnakenberg (void) : morph::RD_Base<Flt>() {}

    //! Perform memory allocations, vector resizes and so on.
    void allocate (void) {
        // Always call allocate() from the base class first.
        morph::RD_Base<Flt>::allocate();
        // Resize and zero-initialise the various containers. Note that the size of a
        // 'vector variable' is given by the number of hexes in the hex grid which is
        // a member of this class (via its parent, RD_Base)
        this->resize_vector_vector_variable (this->a);
        // etc
    }

    //! Initilization and any one-time computations required of the model.
    void init (void)
    {
        this->noiseify_vector_variable (this->a[i], 0.5, 1);
    }

    //! Save the variables.
    void save (void)
    {
        std::stringstream fname;
        fname << this->logpath << "/dat_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        morph::HdfData data(fname.str());
        std::stringstream path;

        path << "/a";
        data.add_contained_vals (path.str().c_str(), this->a[i]);
    }

    /*
     * Computation methods
     */

    void compute_genenet(){}

    //! writeme
    void compute_dai_dt (std::vector<Flt>& A_, std::vector<Flt>& dAdt)
    {
        std::vector<Flt> lapA(this->nhex, 0.0);
        this->compute_laplace (A_, lapA);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            dAdt[h] = this->k1 - (this->k2 * A_[h])
                + (this->k3 * A_[h] * A_[h] * this->B[h]) + this->D_A * lapA[h];
        }
    }

    //! Perform one step in the simulation
    void step (void)
    {
        this->stepCount++;

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
                this->compute_dadt (this->A, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K1[h] = dadt[h] * this->dt;
                    Atst[h] = this->A[h] + K1[h] * 0.5 ;
                }

                // Stage 2
                this->compute_dadt (Atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K2[h] = dadt[h] * this->dt;
                    Atst[h] = this->A[h] + K2[h] * 0.5;
                }

                // Stage 3
                this->compute_dadt (Atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K3[h] = dadt[h] * this->dt;
                    Atst[h] = this->A[h] + K3[h];
                }

                // Stage 4
                this->compute_dadt (Atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K4[h] = dadt[h] * this->dt;
                }

                // Final sum together. This could be incorporated in the for loop for
                // Stage 4, but I've separated it out for pedagogy.
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    A[h] += ((K1[h] + 2.0 * (K2[h] + K3[h]) + K4[h])/(Flt)6.0);
                }
            }
        }
    }

}; // RD_Bool
