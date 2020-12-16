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
 * This file provides the RD class for the third model, which introduces the idea that
 * there is a delay between the binding of a signalling molecules on the cell surface,
 * the consequent production of new gene products in the nucleus, and then the emission
 * of new signalling molecules into the extracellular matrix, at which time they can
 * potentially diffuse/move to other cells.
 *
 * See paper/supp.tex
 *
 * Author: Seb James
 * Date: December 2020
 */

#include <morph/RD_Base.h>
#include <morph/Hex.h>
#include <morph/HdfData.h>
#include <morph/Random.h>
#include <morph/bn/GeneNet.h>
#include <morph/bn/Genome.h>
#include <morph/bn/GradGenome.h>
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
template <typename Flt> struct GaussParams;
template <typename Flt> std::ostream& operator<< (std::ostream&, const GaussParams<Flt>&);
template <typename Flt> struct GaussParams
{
    Flt gain;
    Flt sigma;
    Flt sigmasq;
    Flt x;
    Flt y;
    //! The background activation, to which the Gaussian will be added.
    Flt bg = Flt{0};
    std::string str() const
    {
        std::stringstream ss;
        ss << "Gaussian with gain=" << gain << ", sigma=" << sigma << ", at location (" << x << "," << y  << ")";
        if (bg != Flt{0}) { ss << " with background " << bg; }
        return ss.str();
    }
    //! Overload the stream output operator
    friend std::ostream& operator<< <Flt> (std::ostream& os, const GaussParams<Flt>& gp);
};
template <typename Flt> std::ostream& operator<< (std::ostream& os, const GaussParams<Flt>& gp)
{
    os << gp.str();
    return os;
}

/*!
 * Reaction diffusion system in which the reaction is an NK model (a Boolean gene
 * regulatory network) coupled with a similar genome for the gradient ascending (or
 * descending) behaviour.
 */
template <typename Flt, size_t N, size_t K>
class RD_Bool3 : public morph::RD_Base<Flt>
{
public:
    //! Our gene regulatory network
    morph::bn::GeneNet<N,K> grn;

    //! A genome for running the simulation
    morph::bn::Genome<N,K> genome;

    //! These are the a_i(x,t) variables, where the innermost vector is a single buffer
    //! of length nhex and there are a_delay buffers, to enable nuclear processes [the
    //! function G()] to operate on delayed values of a.
    alignas(alignof(std::vector<std::vector<std::vector<Flt>>>))
    std::vector<std::vector<std::vector<Flt>>> a;

    //! The number of timesteps that the signalling proteins are delayed as they are
    //! 'communicated' into the nucleus in a form suitable to drive/influence the
    //! genetic machinery.
    alignas(size_t) size_t a_delay = 25;

    //! gradient of a
    alignas(alignof(std::vector<std::array<std::vector<Flt>, 2> >))
    std::vector<std::array<std::vector<Flt>, 2> > grad_a;

    //! D_i parameters (diffusion constants)
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> D;
    //! 2D/3d^2
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> twoDover3dd;
    alignas(Flt) Flt twoOver3dd;

    //! alpha_i parameters (decay rates). Fixme: Should be std::array<Flt, N>
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> alpha;

    //! beta_i parameters (accretion rates, when gene is in expressing state)
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> beta;

    //! gamma_i parameters: gradient interaction parameters, gamma_i is the interaction
    //! of gene i with any other gene to which it responds (either climbing or
    //! descending its gradient). gamma_i is probably chosen to have the same value for
    //! all i, for simplicity.
    alignas(alignof(std::vector<Flt>))
    std::vector<Flt> gamma;

    //! Explicit variable for T(a_i)
    alignas(alignof(std::vector<std::vector<Flt> >))
    std::vector<std::vector<Flt> > T;

    //! The function F[G(s),T(a_1),...,T(a_N)]
    alignas(alignof(std::vector<std::vector<Flt> >))
    std::vector<std::vector<Flt> > F;

    //! The state of the gene network at each hex, computed by calling develop() on the
    //! state constructed by looking at the expression levels of each gene, a[i]. There
    //! are s_delay vectors of length nhex so that the function G(s) can be evaluated
    //! based on delayed values of s.
    alignas(alignof(std::vector<std::vector<morph::bn::state_t>>))
    std::vector<std::vector<morph::bn::state_t>> s;

    //! The number of timesteps that the nuclear state is delayed as it is
    //! 'communicated' to the extracellular region as protein products.
    alignas(size_t) size_t s_delay = 25;

    //! J_i(x,t) variables - the "flux current of axonal branches of type i". This is a
    //! vector field.
    alignas(alignof(std::vector<std::array<std::vector<Flt>, 2> >))
    std::vector<std::array<std::vector<Flt>, 2> > J;

    //! Holds the divergence of the J_i(x)s (might be div_a)
    alignas(alignof(std::vector<std::vector<Flt> >))
    std::vector<std::vector<Flt> > divJ;

    //! Genome holding the interactions between N genes for gradient ascending/descending.
    morph::bn::GradGenome<N> grad_genome;

    //! The gene expression threshold
    Flt expression_threshold = 0.5f;

    //! Shape for initialisation of a
    std::multimap<unsigned int, GaussParams<Flt>> initialHumps;

    //! Default constructor
    RD_Bool3() : morph::RD_Base<Flt>() {}

    ~RD_Bool3()
    {
        if constexpr (use_expression_threshold == false) {
            delete this->frng;
        }
    }

    //! Perform memory allocations, vector resizes and so on.
    virtual void allocate()
    {
        // Always call allocate() from the base class first. This allocates HexGrid.
        morph::RD_Base<Flt>::allocate();
        // Resize and zero-initialise the various containers. Note that the size of a
        // 'vector variable' is given by the number of hexes in the hex grid which is
        // a member of this class (via its parent, RD_Base)

        // Resize/setup a
        this->a.resize (N);
        for (size_t i = 0; i < N; ++i) {
            this->a[i].resize (this->a_delay);
            for (size_t j = 0; j < this->a_delay; ++j) {
                this->a[i][j].resize (this->nhex, 0);
            }
        }

        this->resize_vector_vector (this->F, N);
        this->resize_vector_vector (this->T, N);
        this->resize_vector_array_vector (this->grad_a, N);
        this->resize_vector_param (this->alpha, N);
        this->resize_vector_param (this->D, N);
        this->resize_vector_param (this->twoDover3dd, N);
        this->resize_vector_param (this->beta, N);
        this->resize_vector_param (this->gamma, N);

        this->s.resize (this->s_delay);
        for (size_t j = 0; j < this->s_delay; ++j)  {
            this->s[j].resize (this->nhex, 0);
        }

        this->resize_vector_vector (this->divJ, N);
        this->resize_vector_array_vector (this->J, N);

        if constexpr (use_expression_threshold == false) {
            this->frng = new morph::RandNormal<Flt>(0, Flt{0.01});
        }
    }

    //! Initilization and any one-time computations required of the model.
    virtual void init()
    {
        this->zero_vector_vector (this->F, N);
        this->zero_vector_vector (this->T, N);
        this->zero_vector_array_vector (this->grad_a, N);
        this->zero_vector_vector (this->divJ, N);
        this->zero_vector_array_vector (this->J, N);

        this->init_a();

        this->twoOver3dd = Flt{2} / Flt{3} * this->d * this->d;
        for (size_t i = 0; i < N; ++i) {
            this->twoDover3dd[i] = (this->D[i] + this->D[i]) / Flt{3} * this->d * this->d;
        }

        this->genome.randomize();
        this->grad_genome.randomize();
    }

    //! Initialise gene. Do so according to json config? That would be an array of
    //! Gaussians, most simply.
    virtual void init_a()
    {
        for (auto ih : this->initialHumps) {
            unsigned int idx = ih.first;
            if (idx >= N) { continue; }
            // Initialise a[idx]
            std::cout << "Init a[" << idx << "] with params: " << ih.second << "\n";
            for (auto h : this->hg->hexen) {
                Flt dsq = morph::MathAlgo::distance_sq<Flt> ({ih.second.x, ih.second.y}, {h.x, h.y});
                Flt gval = ih.second.gain * std::exp (-dsq / (Flt{2} * ih.second.sigmasq));
                for (size_t j = 0; j < this->a_delay; ++j) {
                    // Add background value
                    this->a[idx][j][h.vi] += ih.second.bg;
                    // And Gaussian
                    this->a[idx][j][h.vi] += gval;
                }
            }
        }
    }

    virtual void save()
    {
        std::stringstream fname;
        fname << this->logpath << "/dat_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        morph::HdfData data(fname.str());
        for (size_t i = 0; i<N; ++i) {
            std::stringstream path;
            path << "/a_" << i;
            data.add_contained_vals (path.str().c_str(), this->a[i][this->stepCount % this->a_delay]);
        }
    }

    //! Compute the sum of variable a[_i][:][current]
    Flt sum_a (size_t _i)
    {
        Flt sum = Flt{0};
        for (size_t h=0; h < this->nhex; ++h) {
            sum += this->a[_i][this->stepCount % this->a_delay][h];
        }
        return sum;
    }

    //! If false, then instead of using expression threshold for computing this->s[h],
    //! make a probabilistic determination of the state s.
    static constexpr bool use_expression_threshold = true;

    morph::RandNormal<Flt>* frng;

    //! Compute inputs for the gene regulatory network, its next developed step (for
    //! each hex) and its outputs, storing these in this->T and this->s.
    virtual void compute_genenet()
    {
        if constexpr (use_expression_threshold == false) {
            // Use this->T to hold random numbers
            for (size_t i = 0; i < N; ++i) {
                this->T[i] = this->frng->get (this->nhex);
                for (unsigned int h=0; h<this->nhex; ++h) {
                    this->T[i][h] += this->expression_threshold;
                }
            }
        }

        // 1. Compute T(a_i) in each hex. In each hex, the state may be different
        for (unsigned int h=0; h<this->nhex; ++h) {
            // Note: this-> omitted for shorter lines in this function
            //std::cout << "Set s["<<h<<"][" << s_buf_next << "] = 0\n";
            this->s[s_buf_next][h] = 0x0;
            // Check each gene (a_delay timesteps ago) to find out if its concentration is above threshold.
            for (size_t i = 0; i < N; ++i) {
                // Set s based on a[i][h]
                if constexpr (use_expression_threshold == true) {
                    this->s[s_buf_next][h] |= (a[i][a_buf_oldest][h] > expression_threshold ? 0x1 : 0x0) << i;
                } else {
                    this->s[s_buf_next][h] |= (a[i][a_buf_oldest][h] > T[i][h] ? 0x1 : 0x0) << i;
                }
                // T is a function that returns the amount by which a is above (or
                // below if negative) the expression threshold. Graphed, even if not
                // used.
                this->T[i][h] = a[i][a_buf_oldest][h] - expression_threshold;
            }
            // Now have computed the current state, see what the next state is for the
            // oldest s in the buffer. grn.develop() is G() in the notes and this line
            // turns s into s':
            this->grn.develop (s[s_buf_oldest][h], this->genome);
        }
    }

    /*!
     * Computes the "flux of gene i" term, J_i(x)
     *
     * To be adapted to implement 'guidance gradients' where
     * grad_genome<>::i_climbs/descends_j(i,j) implies that g climbs the gradient of
     * gene j.
     */
    virtual void compute_divJ (std::vector<Flt>& fa, size_t i)
    {
        // Up to 1+N terms compute divJ; see Eqs. bd2divJ in supp.pdf. Some of the N
        // terms may be 0. Overall the computation time for this system will scale as
        // N^2 (N computations of divJ; 1+N computations here).

#pragma omp parallel for //schedule(static) // This was about 10% faster than schedule(dynamic,50).
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            // 1. The D Del^2 a_i term.
            // Compute the sum around the neighbours
            Flt thesum = -6 * fa[hi];

            thesum += fa[(HAS_NE(hi)  ? NE(hi)  : hi)];
            thesum += fa[(HAS_NNE(hi) ? NNE(hi) : hi)];
            thesum += fa[(HAS_NNW(hi) ? NNW(hi) : hi)];
            thesum += fa[(HAS_NW(hi)  ? NW(hi)  : hi)];
            thesum += fa[(HAS_NSW(hi) ? NSW(hi) : hi)];
            thesum += fa[(HAS_NSE(hi) ? NSE(hi) : hi)];

            // Multiply sum by 2D/3d^2 to give term1, the diffusion term
            this->divJ[i][hi] = this->twoDover3dd[i] * thesum;

            // N terms
            std::array<Flt, N> Nterms;
            for (size_t j = 0; j < N; ++j) {
                Nterms[j] = Flt{0};
                if (this->grad_genome.i_climbs_j (i, j) == true) {
                    Nterms[j] = this->gamma[i];
                } else if (this->grad_genome.i_descends_j (i, j) == true) {
                    Nterms[j] = -this->gamma[i];
                } // else next j

                if (Nterms[j] != Flt{0}) {
                    Flt divaj_sum = -6 * this->a[j][a_buf_cur][hi];
                    divaj_sum += this->a[j][a_buf_cur][(HAS_NE(hi)  ? NE(hi)  : hi)];
                    divaj_sum += this->a[j][a_buf_cur][(HAS_NNE(hi) ? NNE(hi) : hi)];
                    divaj_sum += this->a[j][a_buf_cur][(HAS_NNW(hi) ? NNW(hi) : hi)];
                    divaj_sum += this->a[j][a_buf_cur][(HAS_NW(hi)  ? NW(hi)  : hi)];
                    divaj_sum += this->a[j][a_buf_cur][(HAS_NSW(hi) ? NSW(hi) : hi)];
                    divaj_sum += this->a[j][a_buf_cur][(HAS_NSE(hi) ? NSE(hi) : hi)];
                    // Multiply sum by 2/3d^2 to give term1: a_i div(a_j)
                    Flt ai_divaj = fa[hi] * this->twoOver3dd * divaj_sum;

                    // Now compute grad(a_j) . grad(a_i)
                    Flt aj_dot_ai = (this->grad_a[i][0][hi] * this->grad_a[j][0][hi]) + (this->grad_a[i][1][hi] * this->grad_a[j][1][hi]);

                    this->divJ[i][hi] += ( Nterms[j] * (ai_divaj + aj_dot_ai) );
                }
            }
        }
    }

    //! Compute gradient of all a_i(x) (before calling compute_divJ)
    virtual void compute_grad_a()
    {
        for (size_t i = 0; i < N; ++i) {
            this->spacegrad2D (this->a[i][a_buf_cur], this->grad_a[i]);
        }
    }

    static constexpr bool debug_compute_dadt = false;

    virtual void compute_dadt (const size_t i, std::vector<Flt>& a_, std::vector<Flt>& dadt)
    {
        this->compute_divJ (a_, i);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; ++h) {
            // Note: 'term1' is divJ[i][h], as computed by compute_divJ(), above.
            Flt term2 = - this->alpha[i] * a_[h];
            // s[h][blah] is the delayed state of 'expressingness' for each gene. In this
            // version of the function, F_i = s_i.
            this->F[i][h] = (this->s[s_buf_oldest][h] & 1<<i) ? Flt{1} : Flt{0};
            Flt term3 = this->beta[i] * this->F[i][h];
            dadt[h] = this->divJ[i][h] + term2 + term3;
        }
    }

    //! A transfer function for a. tanh, as it's linear as a->0.
    virtual inline Flt transfer_a (const Flt& _a)
    {
        return (_a > Flt{0} ? std::tanh (_a) : Flt{0});
    }

    //! a buffer index for current timestep
    size_t a_buf_cur = 0;
    //! a buffer index for last timestep
    size_t a_buf_oldest = 0;
    //! a buffer index for last timestep
    size_t a_buf_next = 0;

    //! s buffer index for current timestep
    size_t s_buf_cur = 0;
    //! s buffer index for oldest buffered timestep
    size_t s_buf_oldest = 0;
    //! s buffer index for last timestep
    size_t s_buf_next = 0;

    //! Perform one step in the simulation. Will hope not to need to extend this method.
    virtual void step()
    {
        // Compute buffer indices in one place.
        this->a_buf_cur = this->stepCount % this->a_delay;
        this->s_buf_cur = this->stepCount % this->s_delay;
        this->a_buf_oldest = this->stepCount == 0 ? (this->a_delay - 1) : ((this->stepCount-1) % this->a_delay);
        this->s_buf_oldest = this->stepCount == 0 ? (this->s_delay - 1) : ((this->stepCount-1) % this->s_delay);
        this->stepCount++;
        this->a_buf_next = this->stepCount % this->a_delay;
        this->s_buf_next = this->stepCount % this->s_delay;

        //std::cout << "a_buf_cur = " << a_buf_cur << ", and a_buf_next = " << a_buf_next
        //          << ". a_buf_oldest=" << a_buf_oldest << std::endl;

        this->compute_genenet();

        // The compute_dadt calls will need the gradients of all the gene fields
        this->compute_grad_a();

        for (size_t i=0; i<N; ++i) {

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
                this->compute_dadt (i, this->a[i][a_buf_cur], dadt); // Or maybe a[i][last]?
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K1[h] = dadt[h] * this->dt;
                    atst[h] = this->a[i][a_buf_cur][h] + K1[h] * 0.5 ;
                }

                // Stage 2
                this->compute_dadt (i, atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K2[h] = dadt[h] * this->dt;
                    atst[h] = this->a[i][a_buf_cur][h] + K2[h] * 0.5;
                }

                // Stage 3
                this->compute_dadt (i, atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K3[h] = dadt[h] * this->dt;
                    atst[h] = this->a[i][a_buf_cur][h] + K3[h];
                }

                // Stage 4
                this->compute_dadt (i, atst, dadt);
#pragma omp parallel for
                for (unsigned int h=0; h<this->nhex; ++h) {
                    K4[h] = dadt[h] * this->dt;
                    Flt delta_a = ((K1[h] + 2.0 * (K2[h] + K3[h]) + K4[h])/(Flt)6.0);
                    // Place result in a_buf_next
                    this->a[i][a_buf_next][h] += delta_a;
                    this->a[i][a_buf_next][h] = this->transfer_a (this->a[i][a_buf_next][h]);
                }
            }
        }
    }

}; // RD_Bool3
