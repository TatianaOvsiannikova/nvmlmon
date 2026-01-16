#pragma once

#include <nlohmann/json.hpp>
#include <nvml.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <sys/types.h>
#include "parameter.h"

using monitored_value_map = std::map<std::string, unsigned long long>;
using monitored_average_map = std::map<std::string, double>;
using parameter_list = std::map<std::string, std::string>;

//Map of classes that represent each monitored quantity
struct nvml_process_stats {
  unsigned long long sm_util = 0;           // SM %
  unsigned long long mem_util = 0;           // SM %
  unsigned long long fb_mem_used = 0; // Fb  bytes
  double gpu_mem_used_pct = 0.0;      // GPU %
};


struct nvml_device_info {
  unsigned int index;
  std::string name;
  unsigned long long total_mem = 0;   // total fb 
};

// NVML monitor
class nvmlmon {
 public:
  nvmlmon();
  ~nvmlmon();

  // Update GPU statistics for the given list of PIDs
  void update_stats(const std::vector<pid_t>& pids,
                    const std::string read_path = "");

  // Retrieve metrics in text or JSON-compatible maps
  monitored_value_map const get_text_stats();
  monitored_value_map const get_json_total_stats();
  monitored_average_map const get_json_average_stats(
      unsigned long long elapsed_clock_ticks);

  parameter_list const get_parameter_list();

  void const get_hardware_info(nlohmann::json& hw_json);
  void const get_unit_info(nlohmann::json& unit_json);
 
  bool const is_valid() { return valid; }

 private:
  bool valid = false;
  unsigned int ngpus = 0;
  std::vector<nvml_device_info> devices;
  std::vector<unsigned long long> last_ts_;
  // pid GPU statistics 
  std::map<pid_t,  nvml_process_stats> gpu_stats;

  // GPU total fb memory usage MB
  std::map<unsigned int, double> device_total_fbmem_;

  void init_nvml();
  void shutdown_nvml();
};

