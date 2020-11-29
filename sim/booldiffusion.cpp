/*
 * Boolean Diffusion
 *
 * Author: Seb James
 * Date: Nov 2020
 */

#ifndef FLT
# error "Please define FLT as float or double when compiling (hint: See CMakeLists.txt)"
#endif

#include "rd_bool.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <limits>
#include <chrono>

#ifdef COMPILE_PLOTTING
# include <morph/Visual.h>
# include <morph/HexGridVisual.h>
# include <morph/GraphVisual.h>
# include <morph/ColourMap.h>
# include <morph/VisualDataModel.h>
# include <morph/Scale.h>
# include <morph/Vector.h>
//! Helper function to save PNG images with a suitable name
void savePngs (const std::string& logpath, const std::string& name,
               unsigned int frameN, morph::Visual& v) {
    std::stringstream ff1;
    ff1 << logpath << "/" << name<< "_";
    ff1 << std::setw(7) << std::setfill('0') << frameN;
    ff1 << ".png";
    v.saveImage (ff1.str());
}
// A convenience typedef
typedef morph::VisualDataModel<FLT>* VdmPtr;
typedef morph::VisualDataModel<morph::bn::state_t>* VdmStatePtr;
#endif

#include <morph/tools.h>
#include <morph/Config.h>

// N, K are hard defined at global scope
static constexpr size_t N = NGENES;
static constexpr size_t K = NINPUTS;

//! Globally initialise bn::Random instance pointer
template<> morph::bn::Random<N,K>* morph::bn::Random<N,K>::pInstance = 0;

int main (int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/params.json" << std::endl;
        return 1;
    }
    std::string paramsfile (argv[1]);
    morph::Config conf(paramsfile);
    if (!conf.ready) {
        std::cerr << "Error setting up JSON config: " << conf.emsg << std::endl;
        return 1;
    }

    /*
     * Simulation parameters
     */

    const unsigned int steps = conf.getUInt ("steps", 1000UL);
    if (steps == 0) {
        std::cerr << "Not much point simulating 0 steps! Exiting." << std::endl;
        return 1;
    }
    std::cout << "steps to simulate: " << steps << std::endl;
    // After how many simulation steps should a log of the simulation data be written?
    const unsigned int logevery = conf.getUInt ("logevery", 100UL);
    // If true, write over an existing set of logs
    bool overwrite_logs = conf.getBool ("overwrite_logs", false);

    // Handling of log path requires a few lines of code:
    std::string logpath = conf.getString ("logpath", "fromfilename");
    std::string logbase = "";
    if (logpath == "fromfilename") {
        // Using json filename as logpath
        std::string justfile = paramsfile;
        // Remove trailing .json and leading directories
        std::vector<std::string> pth = morph::Tools::stringToVector (justfile, "/");
        justfile = pth.back();
        morph::Tools::searchReplace (".json", "", justfile);
        // Use logbase as the subdirectory into which this should go
        logbase = conf.getString ("logbase", "logs/");
        if (logbase.back() != '/') { logbase += '/'; }
        logpath = logbase + justfile;
    }
    if (argc == 3) {
        std::string argpath(argv[2]);
        std::cerr << "Overriding the config-given logpath " << logpath << " with " << argpath << std::endl;
        logpath = argpath;
        if (overwrite_logs == true) {
            std::cerr << "WARNING: You set a command line log path.\n"
                      << "       : Note that the parameters config permits the program to OVERWRITE LOG\n"
                      << "       : FILES on each run (\"overwrite_logs\" is set to true)." << std::endl;
        }
    }

    // The length of one timestep
    const FLT dt = static_cast<FLT>(conf.getDouble ("dt", 0.00001));

#ifdef COMPILE_PLOTTING
    // Parameters from the config that apply only to plotting:
    const unsigned int plotevery = conf.getUInt ("plotevery", 10);
    // Should the plots be saved as png images?
    const bool saveplots = conf.getBool ("saveplots", false);
    // If true, then write out the logs in consecutive order numbers,
    // rather than numbers that relate to the simulation timestep.
    const bool vidframes = conf.getBool ("vidframes", false);
    unsigned int framecount = 0;

    // Window width and height
    const unsigned int win_width = conf.getUInt ("win_width", 1600UL);
    //unsigned int win_height_default = static_cast<unsigned int>(0.8824f * (float)win_width);
    unsigned int win_height_default = static_cast<unsigned int>(0.5f * (float)win_width);
    const unsigned int win_height = conf.getUInt ("win_height", win_height_default);

    // Set up the morph::Visual object which provides the visualization scene (and
    // a GLFW window to show it in)
    morph::Visual v1 (win_width, win_height, "Boolean Diffusion");
    // Set a dark blue background (black is the default). This value has the order
    // 'RGBA', though the A(alpha) makes no difference.
    v1.bgcolour = {0.0f, 0.0f, 0.2f, 1.0f};
    // You can lock movement of the scene
    v1.sceneLocked = conf.getBool ("sceneLocked", false);
    v1.scenetrans_stepsize = 0.5;

    // if using plotting, then set up the render clock
    std::chrono::steady_clock::time_point lastrender = std::chrono::steady_clock::now();
#endif

    /*
     * Simulation instantiation
     */

    RD_Bool<FLT, N, K> RD;

    RD.svgpath = ""; // We'll do an elliptical boundary, so set svgpath empty
    RD.ellipse_a = conf.getDouble ("ellipse_a", 0.8);
    RD.ellipse_b = conf.getDouble ("ellipse_b", 0.6);
    RD.logpath = logpath;
    RD.hextohex_d = conf.getFloat ("hextohex_d", 0.01f);
    RD.boundaryFalloffDist = conf.getFloat ("boundaryFalloffDist", 0.01f);
    RD.allocate();
    RD.set_dt (dt);

    // Set the Boolean Diffusion model parameters
    const Json::Value params = conf.getArray ("model_params");
    unsigned int npar = static_cast<unsigned int>(params.size());
    if (npar != N) {
        std::cerr << "Number of parameter sets in config must be N=" << N
                  << " for this compiled instance of the program. Exiting."
                  << std::endl;
        return 1;
    }
    for (unsigned int i = 0; i < N; ++i) {
        Json::Value v = params[i];
        RD.alpha[i] = v.get("alpha", 1.0).asDouble();
        RD.D[i] = v.get("D", 0.01).asDouble();
        RD.Delta[i] = v.get("Delta", 0.1).asDouble();
    }
    RD.expression_threshold = conf.getDouble ("expression_threshold", 0.5f);

    RD.init();
    // After init, genome is randomized. To set from a previous state, do so here.
    // Set the funky genome
    //RD.genome = {0xb646dd22,0x76617edc,0x7046bfaa,0x58da51aa,0x13393d22};
    std::cout << RD.genome.table() << std::endl;

    // Set the steepness of the sigmoid
    RD.k = 1.0f;

    // Create a log directory if necessary, and exit on any failures.
    if (morph::Tools::dirExists (logpath) == false) {
        morph::Tools::createDir (logpath);
        if (morph::Tools::dirExists (logpath) == false) {
            std::cerr << "Failed to create the logpath directory "
                      << logpath << " which does not exist." << std::endl;
            return 1;
        }
    } else {
        // Directory DOES exist. See if it contains a previous run and
        // exit without overwriting to avoid confusion.
        if (overwrite_logs == false
            && (morph::Tools::fileExists (logpath + "/params.json") == true
                || morph::Tools::fileExists (logpath + "/positions.h5") == true)) {
            std::cerr << "Seems like a previous simulation was logged in " << logpath << ".\n"
                      << "Please clean it out manually, choose another directory or set\n"
                      << "overwrite_logs to true in your parameters config JSON file." << std::endl;
            return 1;
        }
    }

    // As RD.allocate() as been called (and log directory has been created/verified
    // ready), positions can be saved to an HDF5 file:
    RD.savePositions();

#ifdef COMPILE_PLOTTING
    // Before starting the simulation, create the HexGridVisuals.

    // Spatial offset, for positioning of HexGridVisuals
    morph::Vector<float> spatOff;
    float yzero = 0.9f;

    // A. Offset in x direction to the left.
    spatOff = { 1.4f, yzero, 0.0 };
    // Z position scaling - how hilly/bumpy the visual will be.
    std::array<unsigned int, N> grids;
    v1.setCurrent();
    for (unsigned int i = 0; i < N; ++i) {
        morph::Scale<FLT> zscale; zscale.setParams (0.2f, 0.0f);
        // The second is the colour scaling. Set this to autoscale.
        morph::Scale<FLT> cscale; cscale.do_autoscale = true;
        morph::HexGridVisual<FLT>* hgv = new morph::HexGridVisual<FLT> (v1.shaderprog, v1.tshaderprog,
                                                                        RD.hg,
                                                                        spatOff,
                                                                        &(RD.a[i]),
                                                                        zscale,
                                                                        cscale,
                                                                        morph::ColourMapType::Jet);
        std::stringstream ss;
        char gc = 'a';
        gc+=i; ss << gc;
        hgv->addLabel (ss.str(), {RD.ellipse_a+0.05f, 0.0f, 0.01f}, morph::colour::white);

        grids[i] = v1.addVisualModel (hgv);
        spatOff[1] -= (3.0f * conf.getFloat ("ellipse_b", 0.8f));
    }
    morph::Vector<float> stateGraph = spatOff;

    spatOff = { 2.2f, yzero, 0.0 };
    std::array<unsigned int, N> overthresh;
    for (unsigned int i = 0; i < N; ++i) {
        morph::Scale<FLT> zscale; zscale.setParams (0.0f, 0.0f);
        // The second is the colour scaling. Set this to autoscale.
        morph::Scale<FLT> cscale; cscale.compute_autoscale (FLT{0}, FLT{1});
        morph::HexGridVisual<FLT>* hgv = new morph::HexGridVisual<FLT> (v1.shaderprog, v1.tshaderprog,
                                                                        RD.hg,
                                                                        spatOff,
                                                                        &(RD.G[i]),
                                                                        zscale,
                                                                        cscale,
                                                                        morph::ColourMapType::Jet);
        std::stringstream ss;
        char gc = 'a';
        gc+=i;
        ss << "(>thres) " << gc;
        hgv->addLabel (ss.str(), {RD.ellipse_a+0.05f, 0.0f, 0.01f}, morph::colour::white);

        overthresh[i] = v1.addVisualModel (hgv);
        spatOff[1] -= (3.0f * conf.getFloat ("ellipse_b", 0.8f));
    }

    spatOff = { 3.0f, yzero, 0.0 };
    std::array<unsigned int, N> expressing;
    for (unsigned int i = 0; i < N; ++i) {
        morph::Scale<FLT> zscale; zscale.setParams (0.0f, 0.0f);
        // The second is the colour scaling. Set this to autoscale.
        morph::Scale<FLT> cscale; cscale.compute_autoscale (FLT{0}, FLT{1});
        morph::HexGridVisual<FLT>* hgv = new morph::HexGridVisual<FLT> (v1.shaderprog, v1.tshaderprog,
                                                                        RD.hg,
                                                                        spatOff,
                                                                        &(RD.H[i]),
                                                                        zscale,
                                                                        cscale,
                                                                        morph::ColourMapType::Jet);
        std::stringstream ss;
        char gc = 'a';
        gc+=i;
        ss << "(expr) " << gc;
        hgv->addLabel (ss.str(), {RD.ellipse_a+0.05f, 0.0f, 0.01f}, morph::colour::white);

        expressing[i] = v1.addVisualModel (hgv);
        spatOff[1] -= (3.0f * conf.getFloat ("ellipse_b", 0.8f));
    }

    morph::Scale<morph::bn::state_t, float> zscale; zscale.setParams (0.2f, 0.0f);
    // What params to set on colour scale to ensure that 0 is min and 2^N is max?
    morph::Scale<morph::bn::state_t, float> cscale;
    cscale.compute_autoscale (0, static_cast<morph::bn::state_t>(1<<N));
    morph::HexGridVisual<morph::bn::state_t>* hgv1 = new morph::HexGridVisual<morph::bn::state_t> (v1.shaderprog, v1.tshaderprog,
                                                                                                   RD.hg,
                                                                                                   stateGraph,
                                                                                                   &(RD.s),
                                                                                                   zscale,
                                                                                                   cscale,
                                                                                                   morph::ColourMapType::Jet);

    hgv1->addLabel ("state", {RD.ellipse_a+0.05f, 0.0f, 0.01f}, morph::colour::white);
    unsigned int grid_state = v1.addVisualModel (hgv1);


    // Graph sum[a(t)] for each a
    // v2.setCurrent();
    spatOff = { 0.0f, 0.0f, 0.0 };
    morph::GraphVisual<FLT>* graph = new morph::GraphVisual<FLT> (v1.shaderprog, v1.tshaderprog, spatOff);
    graph->setdarkbg(); // colours axes and text
    graph->twodimensional = false;
    graph->setlimits (0, steps, 0, 0.2);
    graph->policy = morph::stylepolicy::markers;
    graph->ylabel = "mean(a)";
    graph->xlabel = "Sim time";
    for (unsigned int i = 0; i < N; ++i) {
        // What's the absc and data? absc is time, so 0 to steps. data is as yet unknown.
        std::stringstream ss;
        char gc = 'a';
        gc+=i;
        ss << "Gene " << gc;
        graph->prepdata (ss.str());
    }
    graph->finalize();
    v1.addVisualModel (static_cast<morph::VisualModel*>(graph));

    // Graph to probe hex 0
    int hexidx = 0;
    spatOff = { -1.6f, 0.0f, 0.0 };
    morph::GraphVisual<FLT>* graph2 = new morph::GraphVisual<FLT> (v1.shaderprog, v1.tshaderprog, spatOff);
    graph2->setdarkbg(); // colours axes and text
    graph2->twodimensional = false;
    graph2->setlimits (0, steps, 0, 1.0);
    graph2->policy = morph::stylepolicy::markers;
    graph2->ylabel = "a[gene][0]";
    graph2->xlabel = "Sim time";
    for (unsigned int i = 0; i < N; ++i) {
        // What's the absc and data? absc is time, so 0 to steps. data is as yet unknown.
        std::stringstream ss;
        char gc = 'a';
        gc+=i;
        ss << "Gene " << gc;
        graph2->prepdata (ss.str());
    }
    graph2->finalize();
    v1.addVisualModel (static_cast<morph::VisualModel*>(graph2));

#endif

    /*
     * Run the simulation
     */

    bool finished = false;
    //std::array<std::vector<FLT>, N> sum_a;
    //std::vector<float> simtime;
    while (finished == false) {
        RD.step();
#ifdef COMPILE_PLOTTING
        if ((RD.stepCount % plotevery) == 0) {
            // These two lines update the data for the two hex grids. That leads to
            // the CPU recomputing the OpenGL vertices for the visualizations.
            morph::gl::Util::checkError (__FILE__, __LINE__);
            for (unsigned int i = 0; i < N; ++i) {
                VdmPtr avm = (VdmPtr)v1.getVisualModel (grids[i]);
                avm->updateData (&(RD.a[i])); // First call to updateData.
                std::cout << "a["<<i<<"][0] = " << RD.a[i][0] << std::endl;
                avm->clearAutoscale();
                avm = (VdmPtr)v1.getVisualModel (overthresh[i]);
                avm->updateData (&(RD.G[i]));
                avm = (VdmPtr)v1.getVisualModel (expressing[i]);
                avm->updateData (&(RD.H[i]));
            }

            VdmStatePtr avm = (VdmStatePtr)v1.getVisualModel (grid_state);
            std::cout << "RD.s[0] = " << (unsigned int)RD.s[0]
                      << " = " << morph::bn::GeneNet<N,K>::state_str(RD.s[0]) << std::endl;
            avm->updateData (&(RD.s)); // First call to updateData.

            if (saveplots) {
                if (vidframes) {
                    savePngs (logpath, "booldiffusion", framecount, v1);
                    ++framecount;
                } else {
                    savePngs (logpath, "booldiffusion", RD.stepCount, v1);
                }
            }
        }

        // rendering the graphics. After each simulation step, check if enough time
        // has elapsed for it to be necessary to call v1.render().
        std::chrono::steady_clock::duration sincerender = std::chrono::steady_clock::now() - lastrender;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(sincerender).count() > 17) { // 17 is about 60 Hz
            glfwPollEvents();
            v1.render();
            //v2.render();
            lastrender = std::chrono::steady_clock::now();
        }
#endif
        if ((RD.stepCount % logevery) == 0) {
#ifdef COMPILE_PLOTTING
            // Update the graph of sum(a)
            for (unsigned int i = 0; i < N; ++i) {
                graph->append ((float)RD.stepCount, RD.sum_a(i)/(FLT)RD.nhex, i);
                graph2->append ((float)RD.stepCount, RD.sigmoid(RD.a[i][hexidx]), i);
            }
#endif
            RD.save();
        }
        if (RD.stepCount > steps) { finished = true; }
    }

    /*
     * Save simulation runtime information.
     */

    conf.set ("float_width", (unsigned int)sizeof(FLT));
    std::string tnow = morph::Tools::timeNow();
    conf.set ("sim_ran_at_time", tnow.substr(0,tnow.size()-1));
    conf.set ("hextohex_d", RD.hextohex_d);
    conf.set ("final_genome", RD.genome.str());
    conf.set ("dt", RD.get_dt());
    if (argc > 0) { conf.set("argv0", argv[0]); }
    if (argc > 1) { conf.set("argv1", argv[1]); }
    const std::string paramsCopy = logpath + "/params.json";
    conf.write (paramsCopy);
    if (conf.ready == false) {
        std::cerr << "Warning: Something went wrong writing a copy of the params.json: "
                  << conf.emsg << std::endl;
    }

#ifdef COMPILE_PLOTTING
    std::cout << "Ctrl-c or press x in graphics window to exit.\n";
    v1.keepOpen();
#endif

    return 0;
};
