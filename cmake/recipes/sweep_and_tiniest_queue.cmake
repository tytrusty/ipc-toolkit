if(TARGET STQ::CPU)
    return()
endif()

message(STATUS "Third-party: creating target 'STQ::CPU'")

option(STQ_WITH_CPU  "Enable CPU Implementation"   ON)
option(STQ_WITH_CUDA "Enable CUDA Implementation" OFF) # get this through GPU_CCD

include(FetchContent)
FetchContent_Declare(
    sweep_and_tiniest_queue
    GIT_REPOSITORY https://github.com/dbelgrod/broadphase-gpu.git
    GIT_TAG 44f0c8b31a32ad9dcd683f62061e9061202b5902
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(sweep_and_tiniest_queue)