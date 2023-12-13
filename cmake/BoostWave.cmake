if (NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(Boost COMPONENTS wave)
endif ()

if (NOT Boost_FOUND)
  if (DEPENDENCIES_FORCE_DOWNLOAD)
    message (STATUS "DEPENDENCIES_FORCE_DOWNLOAD is ON. Fetching Boost::format and Boost::wave")
  else ()
    message (STATUS "Fetching Boost::format and Boost::wave")
  endif ()
  include (FetchContent)
  set(boost_dependencies
    algorithm
    align
    array
    assert
    atomic
    bind
    chrono
    concept_check
    config
    container
    container_hash
    conversion
    core
    date_time
    describe
    detail
    endian
    exception
    filesystem
    format
    function
    function_types
    functional
    fusion
    integer
    intrusive
    io
    iterator
    lexical_cast
    move
    mp11
    mpl
    multi_index
    numeric_conversion
    optional
    phoenix
    pool
    predef
    preprocessor
    proto
    range
    ratio
    rational
    regex
    serialization
    smart_ptr
    spirit
    static_assert
    system
    thread
    throw_exception
    tokenizer
    tuple
    type_index
    type_traits
    typeof
    unordered
    utility
    variant
    variant2
    wave
    winapi
)
  foreach(dep ${boost_dependencies})
    FetchContent_Declare(
        Boost_${dep}
        GIT_REPOSITORY https://github.com/boostorg/${dep}.git
        GIT_TAG        boost-1.82.0
    )
    list(APPEND boost_names Boost_${dep})
  endforeach()
  FetchContent_MakeAvailable(${boost_names})
  foreach(dep ${boost_dependencies})
    set_target_properties(boost_${dep} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
endif ()
