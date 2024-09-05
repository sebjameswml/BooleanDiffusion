/*
 * Boolean Diffusion
 *
 * Author: Seb James
 * Date: Nov 2020
 */

#ifndef FLT
# error "Please define FLT as float or double when compiling (hint: See CMakeLists.txt)"
#endif

#if defined BD_MARK2
# include "rd_bool2.h"
#elif defined BD_MARK3
# include "rd_bool3.h"
#else
# include "rd_bool1.h"
#endif

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
# include <morph/vec.h>
//! Helper function to save PNG images with a suitable name
void savePngs (const std::string& logpath, const std::string& name,
               unsigned int frameN, morph::Visual<>& v)
{
    std::stringstream ff1;
    ff1 << logpath << "/" << name<< "_";
    ff1 << std::setw(7) << std::setfill('0') << frameN;
    ff1 << ".png";
    v.saveImage (ff1.str());
}
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
        std::cerr << "Usage: " << argv[0] << " /path/to/params.json [0-0:0-0]" << std::endl;
        return 1;
    }
    std::string paramsfile (argv[1]);
    morph::Config conf(paramsfile);
    if (!conf.ready) {
        std::cerr << "Error setting up JSON config: " << conf.emsg << std::endl;
        return 1;
    }

    // Optional genome parameter
    std::string genome_arg("");
    std::string gradgenome_arg("");
    if (argc > 2) {
        std::string option (argv[2]);
        std::vector<std::string> gs = morph::Tools::stringToVector (option, ":");
        if (gs.size() != N) {
            throw std::runtime_error ("Malformed genome arg. Format: genome:gradgenome");
        }
        genome_arg = gs[0];
        gradgenome_arg = gs[1];
        std::cout << "User specified genome "<< genome_arg << std::endl;
        std::cout << "User specified grad genome "<< gradgenome_arg << std::endl;
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
        logbase = conf.getString ("logbase", "logs/");
        if (logbase.back() != '/') { logbase += '/'; }
        logpath = logbase + argpath;
        if (overwrite_logs == true) {
            std::cerr << "WARNING: You set a command line log path.\n"
                      << "       : Note that the parameters config permits the program to OVERWRITE LOG\n"
                      << "       : FILES on each run (\"overwrite_logs\" is set to true)." << std::endl;
        }
    }

    // Requested genome and gradient genome
    std::string requested_genome = conf.getString ("genome", "");
    if (!genome_arg.empty()) { requested_genome = genome_arg; }
    std::string title_str("");
#if defined BD_MARK2 || defined BD_MARK3
    std::string requested_gradgenome = conf.getString ("grad_genome", "");
    if (!gradgenome_arg.empty()) { requested_gradgenome = gradgenome_arg; }
    if (!requested_genome.empty() && !requested_gradgenome.empty()) {
        title_str = requested_genome + " : " + requested_gradgenome;
    }
#else
    title_str = requested_genome;
#endif

#ifdef COMPILE_PLOTTING
    // Parameters from the config that apply only to plotting:
    const unsigned int plotevery = conf.getUInt ("plotevery", 10);
    // Should the plots be saved as png images?
    const bool saveplots = conf.getBool ("saveplots", false);
    // If true, then write out the logs in consecutive order numbers,
    // rather than numbers that relate to the simulation timestep.
    const bool vidframes = conf.getBool ("vidframes", false);
    unsigned int framecount = 0;
    // Auto-scale colour on the main gene expression maps?
    const bool autoscalecolour = conf.getBool ("autoscalecolour", false);
    // Window width and height
    const unsigned int win_width = conf.getUInt ("win_width", 1920UL);
    unsigned int win_height_default = static_cast<unsigned int>(0.5625f * (float)win_width);
    const unsigned int win_height = conf.getUInt ("win_height", win_height_default);
    // Plot 3D surface maps or 2D maps?
    const bool map3d = conf.getBool ("map3d", true);
    // Set up the morph::Visual object which provides the visualization scene (and
    // a GLFW window to show it in)
    morph::Visual<> v1 (win_width, win_height, title_str);
    // A bit of lighting is useful for 3d graphs
    v1.lightingEffects (map3d);
    // Set a dark blue background (black is the default). This value has the order
    // 'RGBA', though the A(alpha) makes no difference.
    //v1.bgcolour = {0.0f, 0.0f, 0.04f, 1.0f};
    v1.backgroundWhite();
    // You can lock movement of the scene
    v1.sceneLocked = conf.getBool ("sceneLocked", false);
    v1.scenetrans_stepsize = 0.5;
    // Config can tell the program to finish as soon as the sim is done
    v1.readyToFinish = conf.getBool ("finish_asap", false);
    // The title is the genome, so show it.
    v1.showTitle = true;
    // if using plotting, then set up the render clock
    std::chrono::steady_clock::time_point lastrender = std::chrono::steady_clock::now();

    // User can specify to plot ONLY certain frames.
    const nlohmann::json plotonly = conf.get("plotonly");
    for (unsigned int i = 0; i < plotonly.size(); ++i) {
        int po = plotonly[i];
        std::cout << "will plot only: " << po << std::endl;
    }

#endif

    /*
     * Simulation instanciation
     */
#if defined BD_MARK2
    RD_Bool2<FLT, N, K> RD;
#elif defined BD_MARK3
    RD_Bool3<FLT, N, K> RD;
#else
    RD_Bool1<FLT, N, K> RD;
#endif

    RD.svgpath = ""; // We'll do an elliptical boundary, so set svgpath empty
    RD.ellipse_a = conf.getDouble ("ellipse_a", 0.8);
    RD.ellipse_b = conf.getDouble ("ellipse_b", 0.6);
    RD.logpath = logpath;
    RD.hextohex_d = conf.getFloat ("hextohex_d", 0.01f);
    RD.boundaryFalloffDist = conf.getFloat ("boundaryFalloffDist", 0.01f);
#if defined BD_MARK3
    RD.a_delay = conf.getInt ("expression_delay", 40);
    RD.s_delay = conf.getInt ("state_delay", 40);
#endif
    RD.allocate();
    RD.set_dt (static_cast<FLT>(conf.getDouble ("dt", 0.00001))); // The length of one timestep

    // Set the Boolean Diffusion model parameters
    const nlohmann::json params = conf.get("model_params");
    unsigned int npar = static_cast<unsigned int>(params.size());
    if (npar != N) {
        std::cerr << "Number of parameter sets in config must be N=" << N
                  << " for this compiled instance of the program. Exiting."
                  << std::endl;
        return 1;
    }
    for (int i = (N-1); i >= 0; i--) {
        nlohmann::json v = params[i];
        std::cout << "Placing parameters for Gene "
                  << (v.contains("name") ? v["name"].get<std::string>() : std::string("unknown"))
                  << " in vector index " << (N-i-1) << std::endl;
        RD.alpha[N-i-1] = v.contains("alpha") ? v["alpha"].get<FLT>() : FLT{1};
        RD.D[N-i-1] = v.contains("D") ? v["D"].get<FLT>() : FLT{0.01};
        RD.beta[N-i-1] = v.contains("beta") ? v["beta"].get<FLT>() : FLT{0.1};
#if defined BD_MARK2 || defined BD_MARK3
        RD.gamma[N-i-1] = v.contains("gamma") ? v["gamma"].get<FLT>() : FLT{1};
#endif

    }
    RD.expression_threshold = conf.getDouble ("expression_threshold", 0.5f);
#if defined BD_MARK2
    RD.expression_delay = conf.getInt ("expression_delay", 1);
    std::cout << "Expression delay is " << RD.expression_delay << " timesteps\n";
#endif

#if defined BD_MARK2 || defined BD_MARK3
    const nlohmann::json init_a_params = conf.get("init_a");
    npar = static_cast<unsigned int>(init_a_params.size());
    for (unsigned int i = 0; i < npar; ++i) {
        nlohmann::json v = init_a_params[i];
        int idx = v.contains("idx") ? v["idx"].get<int>() : -1;
        if (idx >= 0) {
            GaussParams<FLT> gp;
            gp.gain = v.contains("gain") ? v["gain"].get<FLT>() : FLT{1};
            gp.sigma = v.contains("sigma") ? v["sigma"].get<FLT>() : FLT{1};
            gp.sigmasq = gp.sigma * gp.sigma;
            gp.x = v.contains("x") ? v["x"].get<FLT>() : FLT{0};
            gp.y = v.contains("y") ? v["y"].get<FLT>() : FLT{0};
            gp.bg = v.contains("bg") ? v["bg"].get<FLT>() : FLT{0};
            RD.initialHumps.insert (std::make_pair((unsigned int)idx, gp));
        }
    }
#endif

    RD.init();

    // Add a title label in case the title_str was empty at morph::Visual init
    if (title_str.empty()) {
        title_str = RD.genome.str();
#if defined BD_MARK2 || defined BD_MARK3
        title_str += " : " + RD.grad_genome.str();
#endif
#ifdef COMPILE_PLOTTING
        v1.addLabel (title_str, {0,0,0}, morph::TextFeatures(0.025f));
#endif
    }

    // After init, genome is randomized. To set from a previous state, do so here.
    // Set the funky genome
    //RD.genome = {0xb646dd22,0x76617edc,0x7046bfaa,0x58da51aa,0x13393d22};
    if (!requested_genome.empty()) {
        RD.genome.set (requested_genome);
    }
#if defined BD_MARK2 || defined BD_MARK3
    if (!requested_gradgenome.empty()) {
        RD.grad_genome.set (requested_gradgenome);
    }
    std::cout << "Full genome: " << RD.genome << "::" << RD.grad_genome << std::endl;
    std::cout << "Transcription genome:\n";
    std::cout << RD.genome.table() << std::endl;
    std::cout << "Per-gene tables:\n";
    std::cout << morph::bn::GeneNet<N,K>::gene_tables(RD.genome) << std::endl;
    std::cout << RD.grad_genome.table() << std::endl;
#else
    std::cout << RD.genome.table() << std::endl;
    std::cout << "Per-gene tables:\n";
    std::cout << morph::bn::GeneNet<N,K>::gene_tables(RD.genome) << std::endl;
#endif

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
    // Labels
    // RD.genome.table() is not as good as gene_tables
    std::stringstream gss0;
    gss0 << "Transln genome: " << RD.genome.str() << ":\n\n" << RD.genome.shorttable();
    float x_tabstart = 0.6f;
    float y_tab = -0.32f;
    float tabspace = 0.01f;
    morph::vec<float, 3> tabpos = { x_tabstart, y_tab, 0.0f };
    morph::TextGeometry dims1 = v1.addLabel (gss0.str(), tabpos,
                                             morph::TextFeatures(0.01f, 24, false, morph::colour::black, morph::VisualFont::VeraMono));
    std::cout << "Table 1 has width/height: " << dims1.width() << "/" << dims1.height() << std::endl;

    std::stringstream gss;
    tabpos[0] += dims1.width() + tabspace;
    gss << "\n\nPer-gene tables:\n\n" << RD.grn.gene_tables (RD.genome);
    morph::TextGeometry dims2 = v1.addLabel (gss.str(), tabpos,
                                             morph::TextFeatures(0.01f, 24, false, morph::colour::black, morph::VisualFont::VeraMono));
    std::cout << "Table 2 has width/height: " << dims2.width() << "/" << dims2.height() << std::endl;

# if defined BD_MARK2 || defined BD_MARK3
    tabpos[0] += dims2.width() + 3*tabspace;
    std::stringstream ggss;
    ggss << "Gradient genome: " << RD.grad_genome.str() << "\n\n" << RD.grad_genome.shorttable();
    morph::TextGeometry dims3 = v1.addLabel (ggss.str(), tabpos,
                                             morph::TextFeatures(0.01f, 24, false, morph::colour::black, morph::VisualFont::VeraMono));
    std::cout << "Table 3 has width/height: " << dims3.width() << "/" << dims3.height() << std::endl;
# endif
    // Before starting the simulation, create the HexGridVisuals.

    // Spatial offset, for positioning of HexGridVisuals
    morph::vec<float> spatOff;
    float yzero = 0.9f;

    // A. Offset in x direction to the left.
    spatOff = { 1.4f, yzero, 0.0 };
    // Z position scaling - how hilly/bumpy the visual will be.
    std::array<morph::HexGridVisual<FLT>*, N> grids;
    v1.setContext();
    // The second is the colour scaling. Set this to autoscale the same for all genes
    morph::Scale<FLT> cscale_gene; cscale_gene.compute_autoscale (FLT{0}, FLT{1});
    for (int i = (N-1); i >= 0; i--) {
        morph::Scale<FLT> zscale; zscale.setParams ((map3d ? 0.2f : 0.0f), 0.0f);
        std::cout << "Create HexGridVisual for RD.a["<<i<<"]..." << std::endl;
        auto hgv = std::make_unique<morph::HexGridVisual<FLT>> (RD.hg.get(), spatOff);
        v1.bindmodel (hgv);
#ifdef BD_MARK3
        hgv->setScalarData (&(RD.a[i][RD.a_buf_next]));
        hgv->zScale = zscale;
        hgv->colourScale = cscale_gene;
        hgv->cm.setType (morph::ColourMapType::Jet);
        // What were the last two? // 0.0f, (map3d ? true : false));
#else
        hgv->setScalarData (&(RD.a[i]));
        hgv->zScale = zscale;
        hgv->colourScale = cscale_gene;
        hgv->cm.setType (morph::ColourMapType::Jet);
#endif
        std::stringstream ss;
        // MSB is 'a'
        char gc = 'a';
        gc+=N;
        gc-=(i+1);
        ss << gc;
        hgv->addLabel (ss.str(), {RD.ellipse_a+0.05f, 0.0f, 0.01f},  morph::TextFeatures(0.01f, morph::colour::white));

        hgv->finalize();
        grids[i] = v1.addVisualModel (hgv);
        spatOff[1] -= (3.0f * conf.getFloat ("ellipse_b", 0.8f));
    }
    morph::vec<float> stateGraph = spatOff;

    spatOff = { 2.2f, yzero, 0.0 };
    std::array<morph::HexGridVisual<FLT>*, N> overthresh;
    for (int i = (N-1); i >= 0; i--) {
        morph::Scale<FLT> zscale; zscale.setParams ((map3d ? 0.2f : 0.0f), 0.0f);
        // The second is the colour scaling. Set this to autoscale.
        morph::Scale<FLT> cscale; cscale.compute_autoscale (FLT{-1}, FLT{1});
        auto hgv = std::make_unique<morph::HexGridVisual<FLT>> (RD.hg.get(), spatOff);
        v1.bindmodel (hgv);
        hgv->setScalarData (&(RD.T[i]));
        hgv->zScale = zscale;
        hgv->colourScale = cscale;
        hgv->cm.setType (map3d ? morph::ColourMapType::RainbowZeroBlack : morph::ColourMapType::Jet);
        std::stringstream ss;
        // MSB is 'a'
        char gc = 'a';
        gc+=N;
        gc-=(i+1);
        ss << "T(" << gc << ")";
        hgv->addLabel (ss.str(), {RD.ellipse_a+0.05f, 0.0f, 0.01f}, morph::TextFeatures(0.01f, morph::colour::white));
        hgv->finalize();
        overthresh[i] = v1.addVisualModel (hgv);
        spatOff[1] -= (3.0f * conf.getFloat ("ellipse_b", 0.8f));
    }

    spatOff = { 3.0f, yzero, 0.0 };
    std::array<morph::HexGridVisual<FLT>*, N> expressing;
    for (int i = (N-1); i >= 0; i--) {
        morph::Scale<FLT> zscale; zscale.setParams ((map3d ? 0.02f : 0.0f), 0.0f);
        // The second is the colour scaling. Set this to autoscale.
        morph::Scale<FLT> cscale; cscale.compute_autoscale (FLT{0}, FLT{1});
        auto hgv = std::make_unique<morph::HexGridVisual<FLT>> (RD.hg.get(), spatOff);
        v1.bindmodel (hgv);
        hgv->setScalarData (&(RD.F[i]));
        hgv->zScale = zscale;
        hgv->colourScale = cscale;
        hgv->cm.setType (map3d ? morph::ColourMapType::RainbowZeroBlack : morph::ColourMapType::Jet);
        std::stringstream ss;
        // MSB is 'a'
        char gc = 'a';
        gc+=N;
        gc-=(i+1);
        ss << "F_" << gc << "()";
        hgv->addLabel (ss.str(), {RD.ellipse_a+0.05f, 0.0f, 0.01f}, morph::TextFeatures(0.01f, morph::colour::white));
        hgv->finalize();
        expressing[i] = v1.addVisualModel (hgv);
        spatOff[1] -= (3.0f * conf.getFloat ("ellipse_b", 0.8f));
    }

    morph::Scale<morph::bn::state_t, float> zscale; zscale.setParams (0.0f, 0.0f);
    // What params to set on colour scale to ensure that 0 is min and 2^N is max?
    morph::Scale<morph::bn::state_t, float> cscale;
    cscale.compute_autoscale (0, static_cast<morph::bn::state_t>(1<<N));
    auto hgv1 = std::make_unique<morph::HexGridVisual<morph::bn::state_t>> (RD.hg.get(), stateGraph);
    v1.bindmodel (hgv1);
#ifdef BD_MARK3
    hgv1->setScalarData (&(RD.s[RD.s_buf_next]));
#else
    hgv1->setScalarData (&(RD.s));
#endif
    hgv1->zScale = zscale;
    hgv1->colourScale = cscale;
    hgv1->cm.setType (morph::ColourMapType::Jet);
    hgv1->addLabel ("state", {RD.ellipse_a+0.05f, 0.0f, 0.01f}, morph::TextFeatures(0.01f, morph::colour::white));
    hgv1->finalize();
    auto grid_state = v1.addVisualModel (hgv1);


    // Graph sum[a(t)] for each a
    spatOff = { 0.0f, 0.0f, 0.0 };
    auto graph = std::make_unique<morph::GraphVisual<FLT>> (spatOff);
    v1.bindmodel (graph);
    //graph->setdarkbg(); // colours axes and text
    graph->twodimensional = false;
    graph->setlimits (0, steps, 0, conf.getFloat("graph_mean_ymax", 1.0f));
    graph->policy = morph::stylepolicy::lines; // markers, both or allcolour
    graph->ylabel = "mean(a)";
    graph->xlabel = "Sim time";
    for (int i = N-1; i >= 0; i--) {
        // What's the absc and data? absc is time, so 0 to steps. data is as yet unknown.
        std::stringstream ss;
        // MSB is 'a'
        char gc = 'a';
        gc+=N;
        gc-=(i+1);
        ss << "Gene " << gc;
        // The first one that gets prepdata called for it will be the first one in the list.
        graph->prepdata (ss.str());
    }
    graph->finalize();
    auto graph_ptr = v1.addVisualModel (graph);

    // Graph to probe a single hex
    int hexidx = conf.getInt ("graph_hex", -1);
    int hexri = conf.getInt ("graph_hexri", 0);
    int hexgi = conf.getInt ("graph_hexgi", 0);
    if (hexidx == -1) {
        // Find a hex by ri/gi to graph
        for (auto h : RD.hg->hexen) {
            if (h.ri == hexri && h.gi == hexgi) {
                hexidx = h.vi;
                std::cout << "Hex at r/g = " << hexri << "/" << hexgi << " is hex index " << hexidx << std::endl;
                break;
            }
        }
    }
    if (hexidx == -1) { hexidx = 0; }

    spatOff = { -1.6f, 0.0f, 0.0 };
    auto graph2 = std::make_unique<morph::GraphVisual<FLT>> (spatOff);
    v1.bindmodel (graph2);
    //graph2->setdarkbg(); // colours axes and text
    graph2->twodimensional = false;
    graph2->setlimits (0, steps, 0, conf.getFloat("graph_single_ymax", 1.0f));
    graph2->policy = morph::stylepolicy::lines;
    std::stringstream yy;
    yy << "a[gene][" << hexidx << "]";
    graph2->ylabel = yy.str();
    graph2->xlabel = "Sim time";
    for (int i = N-1; i >= 0; i--) {
        // What's the absc and data? absc is time, so 0 to steps. data is as yet unknown.
        std::stringstream ss;
        // MSB is 'a'
        char gc = 'a';
        gc+=N;
        gc-=(i+1);
        ss << "Gene " << gc;
        graph2->prepdata (ss.str());
    }
    graph2->finalize();
    auto graph2_ptr = v1.addVisualModel (graph2);

    bool pureplot = conf.getBool ("pureplot", false); // pureplot means we're rendering only as necessary for saving pngs
#endif

    /*
     * Run the simulation
     */
    bool finished = false;
    while (finished == false) {
        RD.step();
#ifdef COMPILE_PLOTTING
        bool doplot = false;
        if (plotonly.size()) {
            for (unsigned int i = 0; i < plotonly.size(); ++i) {
                unsigned int po = plotonly[i];
                if (po == RD.stepCount) {
                    doplot = true;
                    break;
                }
            }
        } else {
            doplot = (RD.stepCount % plotevery) == 0;
        }
        if (doplot) {
            // These two lines update the data for the two hex grids. That leads to
            // the CPU recomputing the OpenGL vertices for the visualizations.
            morph::gl::Util::checkError (__FILE__, __LINE__);
            for (unsigned int i = 0; i < N; ++i) {
#ifdef BD_MARK3
                grids[i]->updateData (&(RD.a[i][RD.a_buf_next]));
#else
                grids[i]->updateData (&(RD.a[i]));
#endif
                if (autoscalecolour == true) {
                    grids[i]->clearAutoscale();
                }
                overthresh[i]->updateData (&(RD.T[i]));
                expressing[i]->updateData (&(RD.F[i]));
            }

            //std::cout << "RD.s[0] = " << (unsigned int)RD.s[0]
            //          << " = " << morph::bn::GeneNet<N,K>::state_str(RD.s[0]) << std::endl;
#ifdef BD_MARK3
            grid_state->updateData (&(RD.s[RD.s_buf_next]));
#else
            grid_state->updateData (&(RD.s));
#endif

            if (saveplots) {
                v1.render();
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
        if (!pureplot) {
            std::chrono::steady_clock::duration sincerender = std::chrono::steady_clock::now() - lastrender;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(sincerender).count() > 17) { // 17 is about 60 Hz
                glfwPollEvents();
                v1.render();
                lastrender = std::chrono::steady_clock::now();
            }
        }
#endif
        if ((RD.stepCount % logevery) == 0) {

            // Would need to do this for each and every hex. Then get, for each state
            // hexmap, N 'inputs' hex maps. Do I really want or need that?
            //std::array<state_t, N> inputs;
            //morph::GeneNet<N,K>::setup_inputs (RD.s[?], inputs);

#ifdef COMPILE_PLOTTING
            // Update the graph of sum(a)
            for (unsigned int i = 0; i < N; ++i) {
                // The 0th curve on the graph is the last/MSB gene (i.e. 'a')
                graph_ptr->append ((float)RD.stepCount, RD.sum_a(i)/(FLT)RD.nhex, (N-i-1));
#ifdef BD_MARK3
                graph2_ptr->append ((float)RD.stepCount, RD.a[i][RD.a_buf_next][hexidx], (N-i-1));
#else
                graph2_ptr->append ((float)RD.stepCount, RD.a[i][hexidx], (N-i-1));
#endif
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
    conf.set ("genome_used", RD.genome.str());
#if defined BD_MARK2 || defined BD_MARK3
    conf.set ("grad_genome_used", RD.grad_genome.str());
#endif
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
