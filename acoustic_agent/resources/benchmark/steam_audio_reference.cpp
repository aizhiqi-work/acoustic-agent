#include <phonon.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int option(int argc, char** argv, const std::string& name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == name)
            return std::atoi(argv[i + 1]);
    return fallback;
}

float option(int argc, char** argv, const std::string& name, float fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == name)
            return std::atof(argv[i + 1]);
    return fallback;
}

bool ok(IPLerror status, const char* operation) {
    if (status == IPL_STATUS_SUCCESS)
        return true;
    std::cerr << operation << " failed with status " << status << "\n";
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const int rays = option(argc, argv, "--rays", 8192);
    const int bounces = option(argc, argv, "--bounces", 32);
    const float duration = option(argc, argv, "--duration", 1.2f);
    constexpr float width = 6.0f;
    constexpr float depth = 4.0f;
    constexpr float height = 2.8f;
    constexpr float absorption = 0.25f;
    constexpr int sampleRate = 16000;

#if defined(__aarch64__) || defined(__arm64__)
    constexpr IPLSIMDLevel simdLevel = IPL_SIMDLEVEL_NEON;
#else
    constexpr IPLSIMDLevel simdLevel = IPL_SIMDLEVEL_AVX2;
#endif
    IPLContextSettings contextSettings{STEAMAUDIO_VERSION, nullptr, nullptr, nullptr, simdLevel};
    IPLContext context = nullptr;
    if (!ok(iplContextCreate(&contextSettings, &context), "iplContextCreate"))
        return 2;

    IPLSceneSettings sceneSettings{};
    sceneSettings.type = IPL_SCENETYPE_DEFAULT;
    IPLScene scene = nullptr;
    if (!ok(iplSceneCreate(context, &sceneSettings, &scene), "iplSceneCreate"))
        return 3;

    const IPLVector3 vertices[] = {
        {0, 0, 0}, {width, 0, 0}, {width, 0, depth}, {0, 0, depth},
        {0, height, 0}, {width, height, 0}, {width, height, depth}, {0, height, depth},
    };
    const IPLTriangle triangles[] = {
        {{0, 2, 1}}, {{0, 3, 2}}, {{4, 5, 6}}, {{4, 6, 7}},
        {{0, 1, 5}}, {{0, 5, 4}}, {{1, 2, 6}}, {{1, 6, 5}},
        {{2, 3, 7}}, {{2, 7, 6}}, {{3, 0, 4}}, {{3, 4, 7}},
    };
    std::vector<IPLint32> materialIndices(12, 0);
    IPLMaterial material{};
    material.scattering = 0.05f;
    for (int band = 0; band < IPL_NUM_BANDS; ++band) {
        material.absorption[band] = absorption;
        material.transmission[band] = 0.0f;
    }
    IPLStaticMeshSettings meshSettings{};
    meshSettings.numVertices = 8;
    meshSettings.numTriangles = 12;
    meshSettings.numMaterials = 1;
    meshSettings.vertices = const_cast<IPLVector3*>(vertices);
    meshSettings.triangles = const_cast<IPLTriangle*>(triangles);
    meshSettings.materialIndices = materialIndices.data();
    meshSettings.materials = &material;
    IPLStaticMesh mesh = nullptr;
    if (!ok(iplStaticMeshCreate(scene, &meshSettings, &mesh), "iplStaticMeshCreate"))
        return 4;
    iplStaticMeshAdd(mesh, scene);
    iplSceneCommit(scene);

    IPLSimulationSettings simulationSettings{};
    simulationSettings.flags = IPL_SIMULATIONFLAGS_REFLECTIONS;
    simulationSettings.sceneType = IPL_SCENETYPE_DEFAULT;
    simulationSettings.reflectionType = IPL_REFLECTIONEFFECTTYPE_HYBRID;
    simulationSettings.maxNumRays = rays;
    simulationSettings.numDiffuseSamples = 128;
    simulationSettings.maxDuration = duration;
    simulationSettings.maxOrder = 1;
    simulationSettings.maxNumSources = 1;
    simulationSettings.numThreads = 1;
    simulationSettings.rayBatchSize = 1;
    simulationSettings.samplingRate = sampleRate;
    simulationSettings.frameSize = 1024;
    IPLSimulator simulator = nullptr;
    if (!ok(iplSimulatorCreate(context, &simulationSettings, &simulator), "iplSimulatorCreate"))
        return 5;
    iplSimulatorSetScene(simulator, scene);

    IPLSourceSettings sourceSettings{};
    sourceSettings.flags = IPL_SIMULATIONFLAGS_REFLECTIONS;
    IPLSource source = nullptr;
    if (!ok(iplSourceCreate(simulator, &sourceSettings, &source), "iplSourceCreate"))
        return 6;
    iplSourceAdd(source, simulator);
    iplSimulatorCommit(simulator);

    IPLSimulationSharedInputs shared{};
    shared.listener.origin = {4.7f, 1.4f, 2.9f};
    shared.listener.ahead = {0, 0, -1};
    shared.listener.up = {0, 1, 0};
    shared.listener.right = {1, 0, 0};
    shared.numRays = rays;
    shared.numBounces = bounces;
    shared.duration = duration;
    shared.order = 1;
    shared.irradianceMinDistance = 1.0f;
    iplSimulatorSetSharedInputs(simulator, IPL_SIMULATIONFLAGS_REFLECTIONS, &shared);

    IPLSimulationInputs inputs{};
    inputs.flags = IPL_SIMULATIONFLAGS_REFLECTIONS;
    inputs.source.origin = {1.3f, 1.4f, 1.1f};
    inputs.source.ahead = {0, 0, -1};
    inputs.source.up = {0, 1, 0};
    inputs.source.right = {1, 0, 0};
    for (int band = 0; band < IPL_NUM_BANDS; ++band)
        inputs.reverbScale[band] = 1.0f;
    inputs.hybridReverbTransitionTime = 1.0f;
    inputs.hybridReverbOverlapPercent = 0.25f;
    iplSourceSetInputs(source, IPL_SIMULATIONFLAGS_REFLECTIONS, &inputs);
    iplSimulatorRunReflections(simulator);

    IPLSimulationOutputs outputs{};
    outputs.reflections.type = IPL_REFLECTIONEFFECTTYPE_HYBRID;
    iplSourceGetOutputs(source, IPL_SIMULATIONFLAGS_REFLECTIONS, &outputs);
    std::cout << "{\n"
              << "  \"engine\": \"Steam Audio native SDK\",\n"
              << "  \"size_m\": [" << width << ", " << depth << ", " << height << "],\n"
              << "  \"source_m\": [1.3, 1.1, 1.4],\n"
              << "  \"listener_m\": [4.7, 2.9, 1.4],\n"
              << "  \"absorption\": " << absorption << ",\n"
              << "  \"sample_rate\": " << sampleRate << ",\n"
              << "  \"duration_s\": " << duration << ",\n"
              << "  \"rays\": " << rays << ",\n"
              << "  \"bounces\": " << bounces << ",\n"
              << "  \"band_count\": " << IPL_NUM_BANDS << ",\n"
              << "  \"reverb_times_s\": [";
    for (int band = 0; band < IPL_NUM_BANDS; ++band) {
        if (band > 0)
            std::cout << ", ";
        std::cout << outputs.reflections.reverbTimes[band];
    }
    std::cout << "]\n}\n";

    iplSourceRemove(source, simulator);
    iplSourceRelease(&source);
    iplSimulatorRelease(&simulator);
    iplStaticMeshRelease(&mesh);
    iplSceneRelease(&scene);
    iplContextRelease(&context);
    return 0;
}
