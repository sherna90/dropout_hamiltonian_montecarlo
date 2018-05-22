cmake_minimum_required(VERSION 2.8)
project( tracker )
SET (CMAKE_CXX_COMPILER             "/usr/bin/g++")
SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11 -fopenmp")
#SET (CMAKE_CXX_FLAGS                "-Wall -g -O3 -w -std=c++11")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
find_package( OpenCV REQUIRED)

OPTION (MAKE_HMC "Make HMC" ON)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "src/likelihood/" )
include_directories( "src/utils/" )
include_directories( "/usr/include/eigen3/" "/usr/local/include/eigen3/" )

set(COMMON_SOURCES
	src/utils/c_utils.cpp 
	src/utils/image_generator.cpp 
	src/likelihood/logistic_regression.cpp 
	src/likelihood/CPU_logistic_regression.cpp  
	src/likelihood/multivariate_gaussian.cpp  
	src/libs/eigenmvn/eigenmvn.h
	src/likelihood/hmc.cpp  
    )

set(SOFTMAX_SOURCES
	src/likelihood/softmax_regression.cpp 
	src/likelihood/CPU_softmax_regression.cpp  
    )

#add_executable( test_op src/test_op.cpp ${COMMON_SOURCES}) 
#target_link_libraries( test_op ${OpenCV_LIBS})

#LR

add_executable( test_lr_iris src/test_lr.cpp ${COMMON_SOURCES}) 
target_link_libraries( test_lr_iris ${OpenCV_LIBS})

add_executable( test_lr_gender src/test_lr_gender.cpp ${COMMON_SOURCES}) 
target_link_libraries( test_lr_gender ${OpenCV_LIBS})

#add_executable( test_lr_mars src/test_lr_mars.cpp ${COMMON_SOURCES}) 
#target_link_libraries( test_lr_mars ${OpenCV_LIBS})

#add_executable( test_lr_madelon src/test_lr_madelon.cpp ${COMMON_SOURCES}) 
#target_link_libraries( test_lr_madelon ${OpenCV_LIBS})

#add_executable( test_lr_cancer src/test_lr_cancer.cpp ${COMMON_SOURCES}) 
#target_link_libraries( test_lr_cancer ${OpenCV_LIBS})

#add_executable( test_lr_gisette src/test_lr_gisette.cpp ${COMMON_SOURCES}) 
#target_link_libraries( test_lr_gisette ${OpenCV_LIBS})

#SOFTMAX

add_executable( test_sr_iris src/test_sr.cpp ${COMMON_SOURCES} ${SOFTMAX_SOURCES}) 
target_link_libraries( test_sr_iris ${OpenCV_LIBS})

add_executable( test_sr_age src/test_sr_age.cpp ${COMMON_SOURCES} ${SOFTMAX_SOURCES}) 
target_link_libraries( test_sr_age ${OpenCV_LIBS})

add_executable( test_sr_mnist src/test_sr_mnist.cpp ${COMMON_SOURCES} ${SOFTMAX_SOURCES}) 
target_link_libraries( test_sr_mnist ${OpenCV_LIBS})

#add_executable( test_sr_bird src/test_sr_bird.cpp ${COMMON_SOURCES} ${SOFTMAX_SOURCES}) 
#target_link_libraries( test_sr_bird ${OpenCV_LIBS})

if(MAKE_HMC)
	set(HMC_SOURCES
		src/likelihood/CPU_hmc.cpp  
	   )
	set(MHMC_SOURCES
		src/likelihood/CPU_mhmc.cpp  
	   )
	set(MDHMC_SOURCES
		src/likelihood/CPU_mdhmc.cpp  
	   )
	set(DHMC_SOURCES
		src/likelihood/CPU_dhmc.cpp  
	   )

	#HMC
	add_executable( test_hmc_iris src/test_hmc.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	target_link_libraries( test_hmc_iris ${OpenCV_LIBS})

	#add_executable( test_hmc_cancer src/test_hmc_cancer.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	#target_link_libraries( test_hmc_cancer ${OpenCV_LIBS})

	add_executable( test_hmc_gender src/test_hmc_gender.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	target_link_libraries( test_hmc_gender ${OpenCV_LIBS})

	#add_executable( test_hmc_gisette src/test_hmc_gisette.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	#target_link_libraries( test_hmc_gisette ${OpenCV_LIBS})

	#add_executable( test_hmc_madelon src/test_hmc_madelon.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	#target_link_libraries( test_hmc_madelon ${OpenCV_LIBS})

	#add_executable( test_hmc_mars src/test_hmc_mars.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	#target_link_libraries( test_hmc_mars ${OpenCV_LIBS})

	#add_executable( test_hmc_birds src/test_hmc_birds.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	#target_link_libraries( test_hmc_birds ${OpenCV_LIBS})

	#add_executable( test_hmc_mnist src/test_hmc_mnist.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	#target_link_libraries( test_hmc_mnist ${OpenCV_LIBS})

	#add_executable( test_all_hmc src/test_all_hmc.cpp ${COMMON_SOURCES} ${HMC_SOURCES}) 
	#target_link_libraries( test_all_hmc ${OpenCV_LIBS})

	#DHMC
	add_executable( test_dhmc_gender src/test_dhmc_gender.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	target_link_libraries( test_dhmc_gender ${OpenCV_LIBS})

	#add_executable( test_dhmc_cancer src/test_dhmc_cancer.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	#target_link_libraries( test_dhmc_cancer ${OpenCV_LIBS})

	#add_executable( test_dhmc_gisette src/test_dhmc_gisette.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	#target_link_libraries( test_dhmc_gisette ${OpenCV_LIBS})

	#add_executable( test_dhmc_madelon src/test_dhmc_madelon.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	#target_link_libraries( test_dhmc_madelon ${OpenCV_LIBS})

	add_executable( test_dhmc_iris src/test_dhmc.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	target_link_libraries( test_dhmc_iris ${OpenCV_LIBS})

	#add_executable( test_dhmc_mnist src/test_dhmc_mnist.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	#target_link_libraries( test_dhmc_mnist ${OpenCV_LIBS})

	#add_executable( test_dhmc_mars src/test_dhmc_mars.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	#target_link_libraries( test_dhmc_mars ${OpenCV_LIBS})

	#add_executable( test_all_dhmc src/test_all_dhmc.cpp ${COMMON_SOURCES} ${DHMC_SOURCES}) 
	#target_link_libraries( test_all_dhmc ${OpenCV_LIBS})

	#MHMC
	add_executable( test_mhmc_iris src/test_mhmc.cpp ${COMMON_SOURCES} ${MHMC_SOURCES} ${SOFTMAX_SOURCES}) 
	target_link_libraries( test_mhmc_iris ${OpenCV_LIBS})

	add_executable( test_mhmc_mnist src/test_mhmc_mnist.cpp ${COMMON_SOURCES} ${MHMC_SOURCES} ${SOFTMAX_SOURCES}) 
	target_link_libraries( test_mhmc_mnist ${OpenCV_LIBS})

	#MDHMC
	add_executable( test_mdhmc_iris src/test_mdhmc.cpp ${COMMON_SOURCES} ${MDHMC_SOURCES} ${SOFTMAX_SOURCES}) 
	target_link_libraries( test_mdhmc_iris ${OpenCV_LIBS})

	add_executable( test_mdhmc_mnist src/test_mdhmc_mnist.cpp ${COMMON_SOURCES} ${MDHMC_SOURCES} ${SOFTMAX_SOURCES}) 
	target_link_libraries( test_mdhmc_mnist ${OpenCV_LIBS})

endif()
