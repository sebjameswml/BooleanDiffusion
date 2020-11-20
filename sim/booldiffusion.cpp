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
# include <morph/ColourMap.h>
# include <morph/VisualDataModel.h>
# include <morph/Scale.h>
# include <morph/Vector.h>
//! Helper function to save PNG images with a suitable name
void savePngs (const std::string& logpath, const std::string& name,
               unsigned int frameN, morph::Visual& v) {
    std::stringstream ff1;
    ff1 << logpath << "/" << name<< "_";
    ff1 << std::setw(5) << std::setfill('0') << frameN;
    ff1 << ".png";
    v.saveImage (ff1.str());
}
#endif

#include <morph/tools.h>
#include <morph/Config.h>

int main (int argc, char **argv)
{
    if (argc < 2) {
        cstd::err << "Usage: " << argv[0] << " /path/to/params.json" << std::endl;
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
    const unsigned int win_width = conf.getUInt ("win_width", 1025UL);
    unsigned int win_height_default = static_cast<unsigned int>(0.8824f * (float)win_width);
    const unsigned int win_height = conf.getUInt ("win_height", win_height_default);

    // Set up the morph::Visual object which provides the visualization scene (and
    // a GLFW window to show it in)
    morph::Visual v1 (win_width, win_height, "Boolean Diffusion");
    // Set a dark blue background (black is the default). This value has the order
    // 'RGBA', though the A(alpha) makes no difference.
    v1.bgcolour = {0.0f, 0.0f, 0.2f, 1.0f};
    // You can tweak the near and far clipping planes
    v1.zNear = 0.001;
    v1.zFar = 20;
    // And the field of view of the visual scene.
    v1.fov = 45;
    // You can lock movement of the scene
    v1.sceneLocked = conf.getBool ("sceneLocked", false);
    // You can set the default scene x/y/z offsets
    v1.setZDefault (conf.getFloat ("z_default", -5.0f));
    v1.setSceneTransXY (conf.getFloat ("x_default", 0.0f),
                        conf.getFloat ("y_default", 0.0f));
    // Make this larger to "scroll in and out of the image" faster
    v1.scenetrans_stepsize = 0.5;

    // if using plotting, then set up the render clock
    std::chrono::steady_clock::time_point lastrender = std::chrono::steady_clock::now();
#endif

    /*
     * Simulation instantiation
     */

    RD_Bool<FLT, 5, 5> RD;

    RD.svgpath = ""; // We'll do an elliptical boundary, so set svgpath empty
    RD.ellipse_a = conf.getDouble ("ellipse_a", 0.8);
    RD.ellipse_b = conf.getDouble ("ellipse_b", 0.6);
    RD.logpath = logpath;
    RD.hextohex_d = conf.getFloat ("hextohex_d", 0.01f);
    RD.boundaryFalloffDist = conf.getFloat ("boundaryFalloffDist", 0.01f);
    RD.allocate();
    RD.set_dt (dt);
    // Set the Boolean Diffusion model parameters
    RD.D[i] = conf.getDouble ("D") // FIXME - might need to do a loop in the json for these values
    RD.init();

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
    float xzero = 0.0f;

    // A. Offset in x direction to the left.
    xzero -= 0.5*RD.hg->width();
    spatOff = { xzero, 0.0, 0.0 };
    // Z position scaling - how hilly/bumpy the visual will be.
    morph::Scale<FLT> zscale; zscale.setParams (0.2f, 0.0f);
    // The second is the colour scaling. Set this to autoscale.
    morph::Scale<FLT> cscale; cscale.do_autoscale = true;
    unsigned int Agrid = v1.addVisualModel (new morph::HexGridVisual<FLT> (v1.shaderprog,
                                                                           RD.hg,
                                                                           spatOff,
                                                                           &(RD.A),
                                                                           zscale,
                                                                           cscale,
                                                                           ColourMapType::Plasma));
    // B. Offset in x direction to the right.
    xzero += RD.hg->width();
    spatOff = { xzero, 0.0, 0.0 };
    unsigned int Bgrid = v1.addVisualModel (new morph::HexGridVisual<FLT> (v1.shaderprog,
                                                                           RD.hg,
                                                                           spatOff,
                                                                           &(RD.B),
                                                                           zscale,
                                                                           cscale,
                                                                           ColourMapType::Jet));
#endif

    /*
     * Run the simulation
     */

    bool finished = false;
    while (finished == false) {
        RD.step();
#ifdef COMPILE_PLOTTING
        if ((RD.stepCount % plotevery) == 0) {
            // These two lines update the data for the two hex grids. That leads to
            // the CPU recomputing the OpenGL vertices for the visualizations.
            morph::VisualDataModel<FLT>* avm = (morph::VisualDataModel<FLT>*)v1.getVisualModel (Agrid);
            avm->updateData (&(RD.A));
            avm->clearAutoscaleColour();

            morph::VisualDataModel<FLT>* bvm = (morph::VisualDataModel<FLT>*)v1.getVisualModel (Bgrid);
            bvm->updateData (&(RD.B));
            bvm->clearAutoscaleColour();

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
        if (duration_cast<milliseconds>(sincerender).count() > 17) { // 17 is about 60 Hz
            glfwPollEvents();
            v1.render();
            lastrender = std::chrono::steady_clock::now();
        }
#endif
        if ((RD.stepCount % logevery) == 0) { RD.save(); }
        if (RD.stepCount > steps) { finished = true; }
    }

    /*
     * Save simulation runtime information.
     */

    conf.set ("float_width", (unsigned int)sizeof(FLT));
    std::string tnow = morph::Tools::timeNow();
    conf.set ("sim_ran_at_time", tnow.substr(0,tnow.size()-1));
    conf.set ("hextohex_d", RD.hextohex_d);
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
