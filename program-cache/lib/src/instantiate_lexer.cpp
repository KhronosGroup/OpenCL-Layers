/*
 * Copyright (c) 2023 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

/// @file instantiate_lexer.cpp
/// @brief Explicit template specializations for the Boost::Wave lexer.
///
/// This file is needed on Windows and mac OS. \c std::string_view::iterator is \c char* on other
/// platforms which is already instantiated in Boost::Wave.

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100) // \W4 - unreferenced formal parameter
#pragma warning(disable : 4702) // \W4 - unreachable code
#pragma warning(disable : 4706) // \W4 - assignment within conditional expression
#endif

#include <boost/wave/cpplexer/re2clex/cpp_re2c_lexer.hpp>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <string_view>

template struct boost::wave::cpplexer::new_lexer_gen<std::string_view::iterator>;
