#pragma once

#include "llama.h"
#include <string>

bool identify_inkling_pack(const std::string & path, bool & inkling, std::string & err);
llama_model * load_inkling_vocab(const std::string & path, llama_model_params params, std::string & err);
