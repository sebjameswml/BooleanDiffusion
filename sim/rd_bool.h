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
#include <morph/BezCurvePath.h>
#include <morph/Hex.h>
#include <morph/HdfData.h>
#include <morph/Random.h>

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

    /*!
     * The contour threshold. For contour plotting [see plot_contour()], the field is
     * normalised, then the contour is plotted where the field crosses this threshold.
     */
    alignas(Flt) Flt contour_threshold = 0.5;

    alignas(Flt) Flt aNoiseGain = 0.1;
    alignas(Flt) Flt aInitialOffset = 0.8;

    /*!
     * Noise for Guidance molecules. Note this is a common parameter, and not a
     * per-guidance molecule parameter.
     */
    alignas(Flt) Flt mNoiseGain = Flt{0};
    alignas(Flt) Flt mNoiseSigma = 0.09; // hex to hex d is usually 0.03

    /*!
     * Data containers for summed n, c and a.
     */
    alignas(std::vector<Flt>) std::vector<Flt> v_nsum;
    alignas(std::vector<Flt>) std::vector<Flt> v_csum;
    alignas(std::vector<Flt>) std::vector<Flt> v_asum;

    /*!
     * ALIGNAS REGION ENDS.
     *
     * Below here, there's no need to worry about alignas keywords.
     */

    /*!
     * Sets the function of the guidance molecule method
     */
    std::vector<FieldShape> rhoMethod;

    /*!
     * Modify initial conditions as if FGF had been mis-expressed posteriorly as well
     * as anteriorly and assume that this has the effect of causing axonal ingrowth in
     * a mirrored fashion. Bunch up the locations of the Gaussians used to set inital
     * conditions (along the x axis) and then duplicate.
     *
     * Note that this is the original approach to Fgf8 mis-expression. See
     * jamesdual.cpp and rd_james_dncomp_dual.h (Seb, Nov 2019).
     */
    bool doFgfDuplication = false;

    /*
     * Dirichlet analysis variables
     */

    //! Dirichlet regions
    std::vector<Flt> regions;

    //! The centroids of the regions. key is the "ID" of the region - a Flt between 0
    //! and 1, with values separated by 1/N.
    std::map<Flt, std::pair<Flt, Flt> > reg_centroids;

    //! The area of each region, by Flt ID (area in number of hexes).
    std::map<Flt, int> region_areas;

    //! Dirichlet vertices
    std::list<morph::DirichVtx<Flt>> vertices;
    std::list<morph::DirichDom<Flt>> domains;

    //! Set to true after calling dirichet() and to false after running step().
    bool dirichletComputed = false;

    //! Key-mapped coordinates of experimental barrels
    std::map<std::string, std::pair<float, float>> expt_centroids;

    /*!
     * From the contour information in the SVG, determine experimental barrel identity
     * for each Hex, This is a float between 0 and 1, with -1 meaning that there is no
     * barrel in that hex on the experimental map (it might be inter-barrel tissue).
     */
    std::vector<Flt> expt_barrel_id;

    /*!
     * The areas (in number of hexes) of each of the barrels, by Flt id.
     */
    std::map<Flt, int> expt_areas;

    //! The overall Honda 1983 Dirichlet approximation. 0.003 is a good fit. 0.05 not
    //! so good.
    Flt honda = 0.0;
    std::map<Flt, Flt> honda_arr;

    //! Honda Dirichlet approx for the experimentally supplied barrels
    Flt expt_honda = 0.0;
    std::map<Flt, Flt> expt_honda_arr;

    /*!
     * A metric to determine the difference between the current pattern and the
     * experimentally observed pattern. Based on a sum of centroid distances between
     * expt and sim barrels.
     */
    Flt sos_distances = 0.0;

    /*!
     * The sum of the square of the absolute differences in area (in num hexes)
     * between the experimental and simulated barrel fields.
     */
    Flt area_diff = 0.0;
    std::vector<Flt> area_diffs;

    /*!
     * Another metric to determine the difference between the current pattern and the
     * experimentally observed pattern, this one is based on traced barrel boundaries.
     */
    Flt mapdiff = 0.0;

    /*!
     * Simple constructor; no arguments. Just calls RD_Base constructor
     */
    RD_James (void) : morph::RD_Base<Flt>() {}

    /*!
     * Initialise this vector of vectors with noise. This is a model-specific
     * function.
     *
     * I apply a sigmoid to the boundary hexes, so that the noise drops away towards
     * the edge of the domain.
     *
     * gp is a set of Gaussian shaped _masks_.
     */
    virtual void noiseify_vector_vector (std::vector<std::vector<Flt> >& vv, std::vector<GaussParams<Flt> >& gp)
    {
        for (unsigned int i = 0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                // boundarySigmoid. Jumps sharply (100, larger is sharper) over length
                // scale 0.05 to 1. So if distance from boundary > 0.05, noise has
                // normal value. Close to boundary, noise is less.
                vv[i][h.vi] = morph::Tools::randF<Flt>() * this->aNoiseGain + (this->aInitialOffset * gp[i].gain);
                if (h.distToBoundary > -0.5) { // It's possible that distToBoundary is set to -1.0
                    Flt bSig = 1.0 / ( 1.0 + std::exp (-100.0*(h.distToBoundary-this->boundaryFalloffDist)) );
                    vv[i][h.vi] = vv[i][h.vi] * bSig;
                }
            }
        }
    }

    /*!
     * Similar to the above, but just adds noise to v (with a gain only) to \a vv. Has no boundary sigmoid.
     */
    virtual void addnoise_vector (std::vector<Flt>& v)
    {
        std::cout << "Add noise to vector?...";
        if (this->mNoiseGain == Flt{0}) {
            // No noise
            std::cout << "NO.\n";
            return;
        }
        std::cout << "Yes.\n";

        // First, fill a duplicate vector with noise
        morph::RandUniform<Flt> rng(-this->mNoiseGain/Flt{2}, this->mNoiseGain/Flt{2});
        std::vector<Flt> noise (v.size(), Flt{0});
        for (unsigned int h = 0; h<v.size(); ++h) {
            noise[h] = rng.get();
        }

        // Set up the Gaussian convolution kernel on a circular HexGrid.
        morph::HexGrid kernel(this->hextohex_d, Flt{20}*this->mNoiseSigma, 0, morph::HexDomainShape::Boundary);
        kernel.setCircularBoundary (Flt{6}*this->mNoiseSigma);
        std::vector<Flt> kerneldata (kernel.num(), 0.0f);
        // Once-only parts of the calculation of the Gaussian.
        Flt one_over_sigma_root_2_pi = 1 / this->mNoiseSigma * 2.506628275;
        Flt two_sigma_sq = 2.0f * this->mNoiseSigma * this->mNoiseSigma;
        Flt gsum = 0;
        for (auto& k : kernel.hexen) {
            Flt gauss = (one_over_sigma_root_2_pi * std::exp ( -(k.r*k.r) / two_sigma_sq ));
            kerneldata[k.vi] = gauss;
            gsum += gauss;
        }
        // Renormalise
        for (size_t k = 0; k < kernel.num(); ++k) { kerneldata[k] /= gsum; }

        // A vector for the result
        std::vector<Flt> convolved (v.size(), 0.0f);

        // Call the convolution method from HexGrid:
        this->hg->convolve (kernel, kerneldata, noise, convolved);

        // Now add the noise to the vector:
        for (size_t h = 0; h < v.size(); ++h) {
            v[h] += convolved[h];
        }
    }

    /*!
     * Apply a mask to the noise in a vector of vectors. This masks with a 2D Gaussian
     * for each a (there are N TC type, so for each i in N, apply a different Gaussian
     * mask, probably with the same width, but different centre).
     *
     * This allows me to initialise the system in a more biologically realistic manner.
     */
    void mask_a (std::vector<std::vector<Flt> >& vv, std::vector<GaussParams<Flt> >& gp)
    {
        // Once-only parts of the calculation of the Gaussian.
        Flt root_2_pi = 2.506628275;

        Flt min_x = 1e7;
        Flt max_x = -1e7;
        Flt scale_m = 1.0;
        Flt scale_c = 0.0;
        if (this->doFgfDuplication == true) {
            DBG2 ("doFgfDuplication is true. N=" << this->N);
            // First compute min and max x, for scaling
            for (unsigned int i = 0; i<this->N && i < gp.size(); ++i) {
                if (!(gp[i].sigma > 0.0)) {
                    continue;
                }
                if (gp[i].x > max_x) {
                    max_x = gp[i].x;
                }
                if (gp[i].x < min_x) {
                    min_x = gp[i].x;
                }
            }
            scale_m = max_x / (max_x - min_x);
            scale_c = -min_x * scale_m;
        }

        for (unsigned int i = 0; i<this->N && i < gp.size(); ++i) {

            if (!(gp[i].sigma > 0.0)) {
                continue;
            }
            std::vector<Flt> vv_cpy(vv[i].size());
            if (this->doFgfDuplication == true) {
                // Note: this duplicates the initial branching density
                // distribution. For the guidance gradients, see code elsewhere.
                gp[i].x = scale_c + scale_m * gp[i].x;
                // In this case, narrow sigma:
                gp[i].sigma /= 2.0;

                // Also copy vv[i] so that we can do the mirrored contribution to the
                // initial state
                vv_cpy.assign(vv[i].begin(), vv[i].end());
                DBG ("Copied. vv_cpy[0] = " << vv_cpy[0] << " vv[i][0] = " << vv[i][0]);
            }

            Flt one_over_sigma_root_2_pi = 1 / gp[i].sigma * root_2_pi;
            Flt two_sigma_sq = 2 * gp[i].sigma * gp[i].sigma;

            for (auto h : this->hg->hexen) {

                Flt rx = gp[i].x - h.x;
                Flt ry = gp[i].y - h.y;
                Flt r = std::sqrt (rx*rx + ry*ry);
                // Note that the gain of the gauss (gp[i].gain) has already been
                // applied in noiseify_vector_vector()
                Flt gauss = (one_over_sigma_root_2_pi
                             * std::exp ( static_cast<Flt>(-(r*r))
                                          / two_sigma_sq ));
                vv[i][h.vi] *= gauss;
            }

            if (this->doFgfDuplication == true) {
                DBG ("-1 * gp[i].x = " << (-1 * gp[i].x));
                // Do mirror contribution
                for (auto h : this->hg->hexen) {
                    Flt rx = (-1 * gp[i].x) - h.x;
                    Flt ry = gp[i].y - h.y;
                    Flt r = std::sqrt (rx*rx + ry*ry);
                    Flt gauss = gp[i].gain * (one_over_sigma_root_2_pi
                                              * std::exp ( static_cast<Flt>(-(r*r))
                                                           / two_sigma_sq ));

                    vv[i][h.vi] += vv_cpy[h.vi] * gauss;

                }
            }
        }
    }

    /*!
     * Perform memory allocations, vector resizes and so on.
     */
    virtual void allocate (void)
    {
        morph::RD_Base<Flt>::allocate();
#ifdef USE_USER_SUPPLIED_CIRCLES
        // Copy the list of circles from the ReadCurves object
        this->expt_centroids = this->r.circles;
        // Invert the y axis of these coordinates, just as the y axis is inverted in
        // void morph::HexGrid::setBoundary (const BezCurvePath<Flt>& p)
        for (auto& c : this->expt_centroids) {
            c.second.second = -c.second.second;
        }
#endif
        // Resize and zero-initialise the various containers
        this->resize_vector_vector (this->c, this->N);
        this->resize_vector_vector (this->dc, this->N);
        this->resize_vector_vector (this->a, this->N);
        this->resize_vector_vector (this->betaterm, this->N);
        this->resize_vector_vector (this->alpha_c, this->N);
        this->resize_vector_vector (this->divJ, this->N);
        this->resize_vector_vector_vector (this->divg_over3d, this->N, this->M);

        this->resize_vector_variable (this->n);
        this->resize_vector_vector (this->rho, this->M);

        this->resize_vector_param (this->alpha, this->N);
        this->resize_vector_param (this->beta, this->N);
        this->resize_vector_param (this->group, this->N);
        this->resize_vector_vector_param (this->gamma, this->N, this->M);

        this->resize_vector_array_vector (this->grad_rho, this->M);

        // Resize grad_a and other vector-array-vectors
        this->resize_vector_array_vector (this->grad_a, this->N);
        this->resize_vector_vector_array_vector (this->g, this->N, this->M);
        this->resize_vector_array_vector (this->J, this->N);

        this->resize_vector_variable (this->expt_barrel_id);

        this->area_diffs.resize (this->N, Flt{0});

        // rhomethod is a vector of size M
        this->rhoMethod.resize (this->M);
        for (unsigned int j=0; j<this->M; ++j) {
            // Set up with Sigmoid1D as default
            this->rhoMethod[j] = FieldShape::Sigmoid1D;
        }

        // Initialise alpha, beta
        for (unsigned int i=0; i<this->N; ++i) {
            this->alpha[i] = 3;
            this->beta[i] = 3;
        }
    }

    /*!
     * Initialise variables and parameters. Carry out one-time computations required
     * of the model. This should be able to re-initialise a finished simulation as
     * well as initialise the first time.
     */
    virtual void init (void)
    {
        DBG ("RD_James::init() called");

        this->stepCount = 0;
        this->dirichletComputed = false;

        // Zero c and n and other temporary variables
        this->zero_vector_vector (this->c, this->N);
        //this->zero_vector_vector (this->a); // gets noisified below
        this->zero_vector_vector (this->betaterm, this->N);
        this->zero_vector_vector (this->alpha_c, this->N);
        this->zero_vector_vector (this->divJ, this->N);
        this->zero_vector_vector_vector (this->divg_over3d, this->N, this->M);

        this->zero_vector_variable (this->n);
        this->zero_vector_vector (this->rho, this->M);

        this->zero_vector_array_vector (this->grad_rho, this->M);

        // Resize grad_a and other vector-array-vectors
        this->zero_vector_array_vector (this->grad_a, this->N);
        this->zero_vector_vector_array_vector (this->g, this->N, this->M);
        this->zero_vector_array_vector (this->J, this->N);

        // Init this one to -1:
#pragma omp parallel for
        for (unsigned int h = 0; h<this->nhex; h++) {
            this->expt_barrel_id[h] = (Flt)-1.0f;
        }

        // Initialise a with noise
        //this->zero_vector_vector (this->a, this->N);
        this->noiseify_vector_vector (this->a, this->initmasks);

        // Mask the noise off (set sigmas to 0 to ignore the masking)
        this->mask_a (this->a, this->initmasks);

        // If client code didn't initialise the guidance molecules, then do so
        if (this->guidance_phi.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_phi.push_back(0.0);
            }
        }
        if (this->guidance_width.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_width.push_back(1.0);
            }
        }
        if (this->guidance_width_ortho.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_width_ortho.push_back(1.0);
            }
        }
        if (this->guidance_offset.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_offset.push_back(0.0);
            }
        }
        if (this->guidance_gain.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_gain.push_back(1.0);
            }
        }
        if (this->guidance_time_onset.empty()) {
            for (unsigned int m=0; m<this->M; ++m) {
                this->guidance_time_onset.push_back(0);
            }
        }

        for (unsigned int m=0; m<this->M; ++m) {
            if (this->rhoMethod[m] == FieldShape::Gauss1D) {
                // Construct Gaussian-waves rather than doing the full-Karbowski shebang.
                this->gaussian1D_guidance (m);
                this->addnoise_vector (this->rho[m]);

            } else if (this->rhoMethod[m] == FieldShape::Gauss2D) {
                // Construct 2 dimensional gradients
                this->gaussian2D_guidance (m);
                this->addnoise_vector (this->rho[m]);

            } else if (this->rhoMethod[m] == FieldShape::Exponential1D) {
                // Construct an 'exponential wave'
                this->exponential_guidance (m);
                this->addnoise_vector (this->rho[m]);

            } else if (this->rhoMethod[m] == FieldShape::Sigmoid1D) {
                this->sigmoid_guidance (m);
                this->addnoise_vector (this->rho[m]);

            } else if (this->rhoMethod[m] == FieldShape::Linear1D) {
                this->linear_guidance (m);
                this->addnoise_vector (this->rho[m]);

            } else if (this->rhoMethod[m] == FieldShape::CircLinear2D) {
                this->circlinear_guidance (m);
                this->addnoise_vector (this->rho[m]);
            }
        }

        // Set up the barrel regions
        std::list<morph::BezCurvePath<float>> ers = this->r.getEnclosedRegions();
        for (auto er : ers) {
            std::pair<float, float> regCentroid; // Don't use it for now...
            std::vector<std::list<morph::Hex>::iterator> regHexes = this->hg->getRegion (er, regCentroid);
            std::string idstr("unknown");
            if (er.name.substr(0,3) == "ol_") { // "ol_" for "outline"
                idstr = er.name.substr(3);
            }
            if (idstr != "unknown") {
                DBG ("Barrel " << idstr << "/" << er.name << " contains " << regHexes.size() << " hexes");
                Flt theid = this->tc_name_to_id (idstr);
                this->expt_areas[theid] = static_cast<int>(regHexes.size());
                // This line instead of the code in the ifdef block "USE_USER_SUPPLIED_CIRCLES":
                this->expt_centroids[idstr] = regCentroid;
                for (auto rh : regHexes) {
                    this->expt_barrel_id[rh->vi] = theid;
                }
            }
        }

        // Compute gradients of guidance molecule concentrations once only
        for (unsigned int m = 0; m<this->M; ++m) {
            this->spacegrad2D (this->rho[m], this->grad_rho[m]);
        }

        // Having computed gradients, build this->g; has to be done once only. Note
        // that a sigmoid is applied so that g(x) drops to zero around the boundary of
        // the domain.
        for (unsigned int i=0; i<this->N; ++i) {
            for (auto h : this->hg->hexen) {
                // Sigmoid/logistic fn params: 100 sharpness, 0.02 dist offset from boundary
                Flt bSig = 1.0 / ( 1.0 + std::exp (-100.0*(h.distToBoundary-this->boundaryFalloffDist)) );
                for (unsigned int m = 0; m<this->M; ++m) {
                    this->g[m][i][0][h.vi] += (this->gamma[m][i] * this->grad_rho[m][0][h.vi]) * bSig;
                    this->g[m][i][1][h.vi] += (this->gamma[m][i] * this->grad_rho[m][1][h.vi]) * bSig;
                }
            }
        }

        this->compute_divg_over3d();
    }

protected:
    /*!
     * Given a TC id string \a idstr, look it up in tcnames and find the Flt ID that it
     * corresponds to. Client code should have set up tcnames.
     */
    Flt tc_name_to_id (const std::string& idstr)
    {
        Flt theid = -1.0;
        typename std::map<Flt, std::string>::iterator tcn = this->tcnames.begin();
        while (tcn != this->tcnames.end()) {
            DBG2 ("Compare " << tcn->second << " and " << idstr << "...");
            if (tcn->second == idstr) {
                DBG ("ID string " << idstr << " matches; set theid to " << tcn->first);
                theid = tcn->first;
                break;
            }
            ++tcn;
        }
        return theid;
    }

    /*!
     * Require private setter for d. Slightly different from the base class version.
     */
    void set_d (Flt d_)
    {
        morph::RD_Base<Flt>::set_d (d_);
        this->updateTwoDover3dd();
    }

public:
    /*!
     * Public accessors for D, as it requires another attribute to be updated at the
     * same time.
     */
    void set_D (Flt D_)
    {
        this->D = D_;
        this->updateTwoDover3dd();
    }
    Flt get_D (void)
    {
        return this->D;
    }

protected:
    /*!
     * Compute 2D/3d^2 (and 1/3d^2 too)
     */
    void updateTwoDover3dd (void)
    {
        this->twoDover3dd = (this->D+this->D) / (3*this->d*this->d);
    }

public:
    /*
     * Parameter setter methods
     */

    /*!
     * setGamma for the guidance molecule index m_idx and the TC index n_idx to
     * \a value. If group_m==m_idx, then set this->group[n_idx]=\a value
     */
    int setGamma (unsigned int m_idx, unsigned int n_idx, Flt value, unsigned int group_m = 0)
    {
        if (gamma.size() > m_idx) {
            if (gamma[m_idx].size() > n_idx) {
                // Ok, we can set the value
                this->gamma[m_idx][n_idx] = value;
                if (group_m == m_idx) {
                    this->group[n_idx] = value;
                    this->groupset.insert (value);
                }
            } else {
                std::cerr << "WARNING: DID NOT SET GAMMA (too few TC axon types for n_idx=" << n_idx << ")" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "WARNING: DID NOT SET GAMMA (too few guidance molecules for m_idx=" << m_idx << ")" << std::endl;
            return 2;
        }
        return 0;
    }

    /*
     * HDF5 file saving/loading methods.
     */

    /*!
     * Save the c, a and n variables.
     */
    virtual void save (void)
    {
        std::stringstream fname;
        fname << this->logpath << "/c_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        morph::HdfData data(fname.str());
        for (unsigned int i = 0; i<this->N; ++i) {
            std::stringstream path;
            // The c variables
            path << "/c" << i;
            data.add_contained_vals (path.str().c_str(), this->c[i]);
            // The a variable
            path.str("");
            path.clear();
            path << "/a" << i;
            data.add_contained_vals (path.str().c_str(), this->a[i]);
            // divJ
            path.str("");
            path.clear();
            path << "/j" << i;
            data.add_contained_vals (path.str().c_str(), this->divJ[i]);
        }
        data.add_contained_vals ("/n", this->n);

        // Dirichlet regions here (as same size as n, c etc) Dirichlet vertices in dv.h5.
        data.add_contained_vals ("/dr", this->regions);
    }

    void saveHG (void)
    {
        std::stringstream hgname;
        hgname << this->logpath << "/hexgrid.h5";
        this->hg->save(hgname.str().c_str());
    }

    // Save out the dirichlet domains, their paths, and various statistical measures.
    void saveDirichletDomains (void)
    {
        std::stringstream fname;
        fname << this->logpath << "/dirich_";
        fname.width(5);
        fname.fill('0');
        fname << this->stepCount << ".h5";
        morph::HdfData data(fname.str());
        unsigned int domcount = 0;
        for (auto dom : this->domains) {
            std::stringstream dname;
            dname << "/dom";
            dname.width(3);
            dname.fill('0');
            dname << domcount++;
            dom.save (data, dname.str());
        }
        // Save the overall honda value for the system
        data.add_val ("/honda", this->honda);
        // And individual honda delta_js:
        // We have to make two vectors out of the map.
        std::vector<Flt> ha_keys;
        std::vector<Flt> ha_vals;
        for (auto mi : this->honda_arr) {
            ha_keys.push_back (mi.first);
            ha_vals.push_back (mi.second);
        }
        data.add_contained_vals ("/honda_arr_keys", ha_keys);
        data.add_contained_vals ("/honda_arr_vals", ha_vals);

        // Save sum of square distances
        data.add_val ("/sos_distances", this->sos_distances);
        // Save sum of square of area differences
        data.add_val ("/area_diff", this->area_diff);
        data.add_contained_vals ("/area_diffs", this->area_diffs);
        // The difference between the experimental barrel map and the simulated map
        data.add_val ("/mapdiff", this->mapdiff);
        // Save the region centroids
        std::vector<Flt> keys;
        std::vector<Flt> x_;
        std::vector<Flt> y_;
        // Hopefully, this ensures that we always save N centroids, even if some
        // default to 0,0.
        for (unsigned int i = 0; i<this->N; ++i) {
            Flt k = (Flt)i/this->N;
            keys.push_back (k);
            x_.push_back (this->reg_centroids[k].first);
            y_.push_back (this->reg_centroids[k].second);

        }
        data.add_contained_vals ("/reg_centroids_id", keys);
        data.add_contained_vals ("/reg_centroids_x", x_);
        data.add_contained_vals ("/reg_centroids_y", y_);
        data.add_val ("/N", this->N);
    }

    /*!
     * Save asum, nsum and csum. Call once at end of simulation.
     */
    void savesums (void)
    {
        std::stringstream fname;
        fname << this->logpath << "/sums.h5";
        morph::HdfData data(fname.str());
        data.add_contained_vals ("/csum", this->v_csum);
        data.add_contained_vals ("/asum", this->v_asum);
        data.add_contained_vals ("/nsum", this->v_nsum);
    }

    /*!
     * Save the guidance molecules to a file (guidance.h5)
     *
     * Also save the experimental ID map in this file, as this is something that needs
     * saving once only.
     */
    void saveGuidance (void)
    {
        std::stringstream fname;
        fname << this->logpath << "/guidance.h5";
        morph::HdfData data(fname.str());
        for (unsigned int m = 0; m<this->M; ++m) {
            std::stringstream path;
            path << "/rh" << m;
            std::string pth(path.str());
            data.add_contained_vals (pth.c_str(), this->rho[m]);
            pth[1] = 'g'; pth[2] = 'x';
            data.add_contained_vals (pth.c_str(), this->grad_rho[m][0]);
            pth[2] = 'y';
            data.add_contained_vals (pth.c_str(), this->grad_rho[m][1]);
            for (unsigned int i = 0; i<this->N; ++i) {
                std::stringstream path;
                path << "/divg_" << m << "_" << i;
                std::string pth(path.str());
                data.add_contained_vals (pth.c_str(), this->divg_over3d[m][i]);
            }
        }
        data.add_contained_vals ("/expt_barrel_id", this->expt_barrel_id);
        // Save the Honda Dirichliform measure for the map
        data.add_val ("/expt_honda", this->expt_honda);
    }

    /*
     * Computation methods
     */

    /*!
     * Compute the values of c, the connection density
     */
    virtual void integrate_c (void)
    {
        // 3. Do integration of c
        for (unsigned int i=0; i<this->N; ++i) {

#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                // Note: betaterm used in compute_dci_dt()
                this->betaterm[i][h] = this->beta[i] * this->n[h] * static_cast<Flt>(pow (this->a[i][h], this->k));
            }

            // Runge-Kutta integration for C (or ci)
            std::vector<Flt> qq(this->nhex,0.);
            std::vector<Flt> k1 = this->compute_dci_dt (this->c[i], i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                qq[h] = this->c[i][h] + k1[h] * this->halfdt;
            }

            std::vector<Flt> k2 = this->compute_dci_dt (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                qq[h] = this->c[i][h] + k2[h] * this->halfdt;
            }

            std::vector<Flt> k3 = this->compute_dci_dt (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                qq[h] = this->c[i][h] + k3[h] * this->dt;
            }

            std::vector<Flt> k4 = this->compute_dci_dt (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; h++) {
                this->dc[i][h] = (k1[h]+2. * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
                Flt c_cand = this->c[i][h] + this->dc[i][h];
                // Avoid over-saturating c_i and make sure dc is similarly modified.
                this->dc[i][h] = (c_cand > 1.0) ? (1.0 - this->c[i][h]) : this->dc[i][h];
                this->c[i][h] = (c_cand > 1.0) ? 1.0 : c_cand;
            }
        }
    }

    /*!
     * A possibly normalization-function specific task to carry out once after the sum
     * of a has been computed.
     */
    virtual void sum_a_computation (const unsigned int _i) {}

    /*!
     * The normalization/transfer function with a default no-op implementation.
     */
    virtual inline Flt transfer_a (const Flt& _a, const unsigned int _i)
    {
        Flt a_rtn = _a;
        return a_rtn;
    }

    /*!
     * Compute the values of a, the branching density
     */
    virtual void integrate_a (void)
    {
        // 2. Do integration of a (RK in the 1D model). Involves computing axon
        // branching flux.

        // Pre-compute:
        // 1) The intermediate val alpha_c.
        for (unsigned int i=0; i<this->N; ++i) {
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->alpha_c[i][h] = this->alpha[i] * this->c[i][h];
            }
        }

        // Runge-Kutta: No OMP here - there are only N(<10) loops, which isn't enough
        // to load the threads up.
        for (unsigned int i=0; i<this->N; ++i) {

            // Runge-Kutta integration for A
            std::vector<Flt> qq(this->nhex, 0.0);
            this->compute_divJ (this->a[i], i); // populates divJ[i]

            std::vector<Flt> k1(this->nhex, 0.0);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k1[h] = this->divJ[i][h] - this->dc[i][h];
                qq[h] = this->a[i][h] + k1[h] * this->halfdt;
            }

            std::vector<Flt> k2(this->nhex, 0.0);
            this->compute_divJ (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k2[h] = this->divJ[i][h] - this->dc[i][h];
                qq[h] = this->a[i][h] + k2[h] * this->halfdt;
            }

            std::vector<Flt> k3(this->nhex, 0.0);
            this->compute_divJ (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k3[h] = this->divJ[i][h] - this->dc[i][h];
                qq[h] = this->a[i][h] + k3[h] * this->dt;
            }

            std::vector<Flt> k4(this->nhex, 0.0);
            this->compute_divJ (qq, i);
#pragma omp parallel for
            for (unsigned int h=0; h<this->nhex; ++h) {
                k4[h] = this->divJ[i][h] - this->dc[i][h];
                this->a[i][h] += (k1[h] + 2.0 * (k2[h] + k3[h]) + k4[h]) * this->sixthdt;
            }

            // Do any necessary computation which involves summing a here
            this->sum_a_computation (i);

            // Now apply the transfer function
//#define DEBUG_SUM_A_TRANSFERRED 1
#ifdef DEBUG_SUM_A_TRANSFERRED
            Flt sum_a_transferred = 0.0;
#endif
#ifndef DEBUG_SUM_A_TRANSFERRED
# pragma omp parallel for
#endif
            for (unsigned int h=0; h<this->nhex; ++h) {
                this->a[i][h] = this->transfer_a (this->a[i][h], i);
#ifdef DEBUG_SUM_A_TRANSFERRED
                sum_a_transferred += this->a[i][h];
#endif
            }
#ifdef DEBUG_SUM_A_TRANSFERRED
            std::cout << "After transfer_a(), sum_a is " << sum_a_transferred << std::endl;
#endif
        }
    }

    /*!
     * Compute n
     */
    virtual void compute_n (void)
    {
        Flt nsum = 0.0;
        Flt csum = 0.0;
#pragma omp parallel for reduction(+:nsum,csum)
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->n[hi] = 0;
            // First, use n[hi] so sum c over all i:
            for (unsigned int i=0; i<this->N; ++i) {
                this->n[hi] += this->c[i][hi];
            }
            // Prevent sum of c being too large:
            this->n[hi] = (this->n[hi] > 1.0) ? 1.0 : this->n[hi];
            csum += this->c[0][hi];
            // Now compute n for real:
            this->n[hi] = 1. - this->n[hi];
            nsum += this->n[hi];
        }

#ifdef DEBUG__
        if (this->stepCount % 100 == 0) {
            DBG ("System computed " << this->stepCount << " times so far...");
            DBG ("sum of n+c is " << nsum+csum);
        }
#endif
    }

    /*!
     * Do a single step through the model.
     */
    virtual void step (void)
    {
        this->stepCount++;

        // 1. Compute Karb2004 Eq 3. (coupling between connections made by each TC type)
        this->compute_n();

        // 2. Call Runge Kutta numerical integration code
        this->integrate_a();
        this->integrate_c();

        this->dirichletComputed = false;
    }

    /*!
     * Examine the value in each Hex of the hexgrid of the scalar field f. If
     * abs(f[h]) exceeds the size of dangerThresh, then output debugging information.
     */
    void debug_values (std::vector<Flt>& f, Flt dangerThresh)
    {
        for (auto h : this->hg->hexen) {
            if (std::abs(f[h.vi]) > dangerThresh) {
                DBG ("Blow-up threshold exceeded at Hex.vi=" << h.vi << " ("<< h.ri <<","<< h.gi <<")" <<  ": " << f[h.vi]);
                unsigned int wait = 0;
                while (wait++ < 120) {
                    usleep (1000000);
                }
            }
        }
    }

    /*!
     * Does: f = (alpha * f) + betaterm. c.f. Karb2004, Eq 1. f is c[i] or q from the
     * RK algorithm.
     */
    std::vector<Flt> compute_dci_dt (std::vector<Flt>& f, unsigned int i)
    {
        std::vector<Flt> dci_dt (this->nhex, 0.0);
#pragma omp parallel for
        for (unsigned int h=0; h<this->nhex; h++) {
            dci_dt[h] = (this->betaterm[i][h] - this->alpha[i] * f[h]);
        }
        return dci_dt;
    }

    /*!
     * Compute the divergence of g and divide by 3d. Used in computation of term2 in
     * compute_divJ().
     *
     * This computation is based on Gauss's theorem.
     */
    void compute_divg_over3d (void)
    {
        // Change to have one for each m in M? They should then sum, right?

        for (unsigned int i = 0; i < this->N; ++i) {

#pragma omp parallel for schedule(static)
            for (unsigned int hi=0; hi<this->nhex; ++hi) {

                std::vector<Flt> divg(this->M, 0.0);
                // Sum up over each gradient.
                for (unsigned int m = 0; m<this->M; ++m) {
                    // First sum
                    if (HAS_NE(hi)) {
                        divg[m] += /*cos (0)*/ (this->g[m][i][0][NE(hi)] + this->g[m][i][0][hi]);
                    } else {
                        // Boundary condition _should_ be satisfied by sigmoidal roll-off of g
                        // towards the boundary, so add only g[i][0][hi]
                        divg[m] += /*cos (0)*/ (this->g[m][i][0][hi]);
                    }
                    if (HAS_NNE(hi)) {
                        divg[m] += /*cos (60)*/ 0.5 * (this->g[m][i][0][NNE(hi)] + this->g[m][i][0][hi])
                            +  (/*sin (60)*/ this->R3_OVER_2 * (this->g[m][i][1][NNE(hi)] + this->g[m][i][1][hi]));
                    } else {
                        //divg += /*cos (60)*/ (0.5 * (this->g[i][0][hi]))
                        //    +  (/*sin (60)*/ this->R3_OVER_2 * (this->g[i][1][hi]));
                    }
                    if (HAS_NNW(hi)) {
                        divg[m] += -(/*cos (120)*/ 0.5 * (this->g[m][i][0][NNW(hi)] + this->g[m][i][0][hi]))
                            +    (/*sin (120)*/ this->R3_OVER_2 * (this->g[m][i][1][NNW(hi)] + this->g[m][i][1][hi]));
                    } else {
                        //divg += -(/*cos (120)*/ 0.5 * (this->g[i][0][hi]))
                        //    +    (/*sin (120)*/ this->R3_OVER_2 * (this->g[i][1][hi]));
                    }
                    if (HAS_NW(hi)) {
                        divg[m] -= /*cos (180)*/ (this->g[m][i][0][NW(hi)] + this->g[m][i][0][hi]);
                    } else {
                        divg[m] -= /*cos (180)*/ (this->g[m][i][0][hi]);
                    }
                    if (HAS_NSW(hi)) {
                        divg[m] -= (/*cos (240)*/ 0.5 * (this->g[m][i][0][NSW(hi)] + this->g[m][i][0][hi])
                                 + ( /*sin (240)*/ this->R3_OVER_2 * (this->g[m][i][1][NSW(hi)] + this->g[m][i][1][hi])));
                    } else {
                        divg[m] -= (/*cos (240)*/ 0.5 * (this->g[m][i][0][hi])
                                 + (/*sin (240)*/ this->R3_OVER_2 * (this->g[m][i][1][hi])));
                    }
                    if (HAS_NSE(hi)) {
                        divg[m] += /*cos (300)*/ 0.5 * (this->g[m][i][0][NSE(hi)] + this->g[m][i][0][hi])
                            - ( /*sin (300)*/ this->R3_OVER_2 * (this->g[m][i][1][NSE(hi)] + this->g[m][i][1][hi]));
                    } else {
                        divg[m] += /*cos (300)*/ 0.5 * (this->g[m][i][0][hi])
                            - ( /*sin (300)*/ this->R3_OVER_2 * (this->g[m][i][1][hi]));
                    }

                    this->divg_over3d[m][i][hi] = divg[m] * this->oneover3d;
                }
            }
        }
    }

    /*!
     * Computes the "flux of axonal branches" term, J_i(x) (Eq 4)
     *
     * Inputs: this->g, fa (which is this->a[i] or a q in the RK algorithm), this->D,
     * i, the TC type.  Helper functions: spacegrad2D().  Output: this->divJ
     *
     * Stable with dt = 0.0001;
     */
    virtual void compute_divJ (std::vector<Flt>& fa, unsigned int i)
    {
        // Compute gradient of a_i(x), for use computing the third term, below.
        this->spacegrad2D (fa, this->grad_a[i]);

        // Three terms to compute; see Eq. 17 in methods_notes.pdf
#pragma omp parallel for //schedule(static) // This was about 10% faster than schedule(dynamic,50).
        for (unsigned int hi=0; hi<this->nhex; ++hi) {

            // 1. The D Del^2 a_i term. Eq. 18.
            // Compute the sum around the neighbours
            Flt thesum = -6 * fa[hi];

            thesum += fa[(HAS_NE(hi)  ? NE(hi)  : hi)];
            thesum += fa[(HAS_NNE(hi) ? NNE(hi) : hi)];
            thesum += fa[(HAS_NNW(hi) ? NNW(hi) : hi)];
            thesum += fa[(HAS_NW(hi)  ? NW(hi)  : hi)];
            thesum += fa[(HAS_NSW(hi) ? NSW(hi) : hi)];
            thesum += fa[(HAS_NSE(hi) ? NSE(hi) : hi)];

            // Multiply sum by 2D/3d^2 to give term1
            Flt term1 = this->twoDover3dd * thesum;

            // 2. The (a div(g)) term.
            Flt term2 = 0.0;

            // 3. Third term is this->g . grad a_i. Should not contribute to J, as
            // g(x) decays towards boundary.
            Flt term3 = 0.0;

            for (unsigned int m =0 ; m < this->M; ++m) {
                if (this->stepCount >= this->guidance_time_onset[m]) {
                    // g contributes to term2
                    term2 += fa[hi] * this->divg_over3d[m][i][hi];
                    // and to term3
                    term3 += this->g[m][i][0][hi] * this->grad_a[i][0][hi] + (this->g[m][i][1][hi] * this->grad_a[i][1][hi]);
                }
            }

            this->divJ[i][hi] = term1 - term2 - term3;
        }
    }

    /*!
     * Generate Gaussian profiles for the chemo-attractants.
     *
     * Instead of using the Karbowski equations, just make some gaussian 'waves'
     *
     * \param m The molecule id
     */
    void gaussian1D_guidance (unsigned int m)
    {
        for (auto h : this->hg->hexen) {
            Flt cosphi = (Flt) std::cos (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt sinphi = (Flt) std::sin (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            DBG2 ("phi: " << guidance_phi[m] << " degrees");
            Flt x_ = (h.x * cosphi) + (h.y * sinphi);
            this->rho[m][h.vi] = guidance_gain[m] * std::exp(-((x_-guidance_offset[m])*(x_-guidance_offset[m])) / guidance_width[m]);
        }
    }

    /*!
     * Circular symmetric 2D Gaussian
     *
     * \param m The molecule id
     */
    void gaussian2D_guidance (unsigned int m)
    {
        // Centre of the Gaussian is offset from 0 by guidance_offset, then rotated by
        // guidance_phi
        Flt x_ = (Flt)this->guidance_offset[m];
        Flt y_ = (Flt)0.0;

        // Rotate the initial location of the 2D Gaussian
        Flt cosphi = (Flt) std::cos (this->TWOPI_OVER_360 * this->guidance_phi[m]);
        Flt sinphi = (Flt) std::sin (this->TWOPI_OVER_360 * this->guidance_phi[m]);
        Flt x_gCentre = (x_ * cosphi) + (y_ * sinphi);
        Flt y_gCentre = - (x_ * sinphi) + (y_ * cosphi);

        for (auto h : this->hg->hexen) {

            Flt rx = x_gCentre - h.x;
            Flt ry = y_gCentre - h.y;
            Flt r = std::sqrt (rx*rx + ry*ry);
            this->rho[m][h.vi] = guidance_gain[m] * std::exp (static_cast<Flt>( -(r*r) / (2.0 * guidance_width[m])) );
        }
    }

    /*!
     * An exponential wave
     *
     * \param m The molecule id
     */
    void exponential_guidance (unsigned int m)
    {
        for (auto h : this->hg->hexen) {
            Flt cosphi = (Flt) std::cos (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt sinphi = (Flt) std::sin (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt x_ = (h.x * cosphi) + (h.y * sinphi);
            this->rho[m][h.vi] = std::exp (this->guidance_gain[m] * (x_-guidance_offset[m]));
        }
    }

    /*!
     * \param m The molecule id
     */
    void sigmoid_guidance (unsigned int m)
    {
        for (auto h : this->hg->hexen) {
            Flt cosphi = (Flt) cos (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt sinphi = (Flt) sin (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            DBG2 ("phi= " << this->guidance_phi[m] << ". cosphi: " << cosphi << " sinphi: " << sinphi);
            Flt x_ = (h.x * cosphi) + (h.y * sinphi);
            DBG2 ("x_[" << h.vi << "] = " << x_);
            this->rho[m][h.vi] = guidance_gain[m] / (1.0 + exp(-(x_-guidance_offset[m])/this->guidance_width[m]));
        }
    }

    /*!
     * \param m The molecule id
     */
    void linear_guidance (unsigned int m)
    {
        std::cout << "Apply linear guidance to molecule " << m << "\n";
        Flt themax = -1e7;
        Flt themin = 1e7;
        for (auto h : this->hg->hexen) {
            Flt cosphi = (Flt) cos (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt sinphi = (Flt) sin (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt x_ = 0.0;
            if (this->doFgfDuplication == true) {
                x_ = (std::abs(h.x) * cosphi) + (h.y * sinphi);
            } else {
                x_ = (h.x * cosphi) + (h.y * sinphi);
            }
            Flt scaled = (x_-guidance_offset[m]) * this->guidance_gain[m];
            this->rho[m][h.vi] = scaled;
            if (scaled > themax) { themax = scaled; }
            if (scaled < themin) { themin = scaled; }
        }
        std::cout << "Linear guidance range was " << themin << " to " << themax << "\n";
    }

    /*!
     * \param m The molecule id
     */
    void circlinear_guidance (unsigned int m)
    {
        for (auto h : this->hg->hexen) {
            // Initial position is guidance_offset * cosphi/sinphi
            Flt cosphi = (Flt) cos (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt sinphi = (Flt) sin (this->TWOPI_OVER_360 * this->guidance_phi[m]);
            Flt x_centre = guidance_offset[m] * cosphi;
            Flt y_centre = guidance_offset[m] * sinphi;

            Flt x_ = (h.x - x_centre);
            Flt y_ = (h.y - y_centre);
            Flt r_ = std::sqrt(x_*x_ + y_*y_);
            this->rho[m][h.vi] = (this->guidance_gain[m] - r_) * this->guidance_gain[m];
        }
    }

    /*!
     * Compute experimental pattern Dirichlet metric
     */
    void expt_dirichlet (void)
    {
        //expt_barrel_id is 'regions'
        this->vertices.clear();
        // Find the vertices and construct domains
        this->domains = morph::ShapeAnalysis<Flt>::dirichlet_vertices (this->hg, this->expt_barrel_id, this->vertices);
        // Carry out the analysis.
        std::vector<std::pair<float, float>> d_centres;
        this->expt_honda = morph::ShapeAnalysis<float>::dirichlet_analyse (this->domains, d_centres, this->expt_honda_arr);
        DBG ("Real barrels have Honda: " << this->expt_honda);
    }

    /*!
     * Compute Dirichlet analysis on the c variable
     */
    void dirichlet (void)
    {
        // Don't recompute unnecessarily
        if (this->dirichletComputed == true) {
            DBG ("dirichlet already computed, no need to recompute.");
            return;
        }

        // Clear out previous results from an earlier timestep
        this->regions.clear();
        this->vertices.clear();
        // Find regions. Based on an 'ID field'.
        this->regions = morph::ShapeAnalysis<Flt>::dirichlet_regions (this->hg, this->c);
        // Compute centroids of regions; used to determine aligned-ness of the barrels
        this->reg_centroids = morph::ShapeAnalysis<Flt>::region_centroids (this->hg, this->regions);

        // based on the reg_centroids, find the sum of squared distances between each
        // simulated barrel and it's experimentally determined location.
        this->sos_distances = 0.0;
        for (unsigned int i = 0; i < this->N; ++i) {
            Flt idx = (Flt)i/(Flt)this->N;
            Flt dx = this->reg_centroids[idx].first
                - (this->expt_centroids[tcnames[idx]].first
                   - this->hg->originalBoundaryCentroid.first);
            Flt dy = this->reg_centroids[idx].second
                - (this->expt_centroids[tcnames[idx]].second
                   - this->hg->originalBoundaryCentroid.second);
            Flt dsq = dx*dx + dy*dy;
#if 0
            DBG2 ("For barrel ID " << tcnames[idx] << ", the sim has centroid locn ("
                  << this->reg_centroids[idx].first << ","
                  << this->reg_centroids[idx].second << ") "
                  << "to compare with expt ("
                  << this->identified_coords[tcnames[idx]].first << "-"
                  <<  this->hg->originalBoundaryCentroid.first << ","
                  << this->identified_coords[tcnames[idx]].second << "-"
                  <<  this->hg->originalBoundaryCentroid.second
                  <<  ") which adds to sos_distances: " << dsq);
#endif
#if 0
            std::cout << tcnames[idx] << ","
                      << this->reg_centroids[idx].first << ","
                      << this->reg_centroids[idx].second << ","
                      << this->expt_centroids[tcnames[idx]].first << ","
                      << this->expt_centroids[tcnames[idx]].second << ","
                      << this->hg->originalBoundaryCentroid.first <<  ","
                      << this->hg->originalBoundaryCentroid.second << std::endl;
#endif
            this->sos_distances += dsq;
        }
        DBG ("overall sos_distances = " << this->sos_distances);

        this->area_diff = 0.0;
        // What's the area of each identified region?
        for (unsigned int i = 0; i < this->N; ++i) {
            Flt idx = (Flt)i/(Flt)this->N;
            int acount = 0;
            for (unsigned int h = 0; h < this->nhex; ++h) {
                acount += this->regions[h] == idx ? 1 : 0;
            }
            this->region_areas[idx] = acount;
            this->area_diffs[i] = std::abs(static_cast<Flt>(this->region_areas[idx] - this->expt_areas[idx]));
            this->area_diff += this->area_diffs[i];
        }
        DBG ("overall area_diff = " << this->area_diff);

        // Compute differences based on barrel ID map. Where hexes are differen
        // between the maps, add 1.0 to the metric; if expt barrel is unset, add 0.5.
        this->mapdiff = 0.0;
        for (unsigned int h = 0; h < this->nhex; ++h) {
            if (this->expt_barrel_id[h] == (Flt)-1.0) {
                this->mapdiff += (Flt)0.5;
            } else {
                this->mapdiff += (this->expt_barrel_id[h] == this->regions[h]) ? 0.0 : 1.0;
            }
        }
        // Normalize by the number of hexes:
        this->mapdiff = this->mapdiff / (Flt)this->nhex;
        DBG ("overall mapdiff = " << this->mapdiff);

        // Find the vertices and construct domains
        this->domains = morph::ShapeAnalysis<Flt>::dirichlet_vertices (this->hg, this->regions, this->vertices);
        // Carry out the analysis.
        std::vector<std::pair<Flt, Flt>> d_centres;
        this->honda = morph::ShapeAnalysis<Flt>::dirichlet_analyse (this->domains, d_centres, this->honda_arr);

        this->dirichletComputed = true;
    }

}; // RD_James
