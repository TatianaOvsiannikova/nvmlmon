#include "nvmlmon.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>

nvmlmon::nvmlmon() {
  init_nvml();
}

nvmlmon::~nvmlmon() {
  shutdown_nvml();
}

void nvmlmon::init_nvml() {
  nvmlReturn_t result = nvmlInit_v2();
  if (result != NVML_SUCCESS) {
    std::cerr << "NVML initialization failed: " << nvmlErrorString(result) << "\n";
    valid = false;
    return;
  }

  if (nvmlDeviceGetCount_v2(&ngpus) != NVML_SUCCESS || ngpus == 0) {
    std::cerr << "No NVIDIA GPU detected.\n";
    nvmlShutdown();
    valid = false;
    return;
  }

  for (unsigned int i = 0; i < ngpus; ++i) {
    nvmlDevice_t handle;
    if (nvmlDeviceGetHandleByIndex_v2(i, &handle) != NVML_SUCCESS) continue;

    nvml_device_info info;
    info.index = i;
    char name_buf[96];
    if (nvmlDeviceGetName(handle, name_buf, sizeof(name_buf)) == NVML_SUCCESS)
      info.name = name_buf;

    nvmlMemory_t memInfo{};
    if (nvmlDeviceGetMemoryInfo(handle, &memInfo) == NVML_SUCCESS)
      info.total_mem = memInfo.total;

    devices.push_back(info);
  }

  valid = true;
}

void nvmlmon::shutdown_nvml() {
  if (valid) nvmlShutdown();
}

void nvmlmon::update_stats(const std::vector<pid_t>& pids, const std::string read_path) {
  if (!valid) return;

  gpu_stats.clear();
  device_total_fbmem_.clear();

  // Optional test mode for unit testing
  if (!read_path.empty()) {
    return;
  }

  for (unsigned int i = 0; i < ngpus; ++i) {
    nvmlDevice_t handle;
    if (nvmlDeviceGetHandleByIndex_v2(i, &handle) != NVML_SUCCESS) continue;

    nvmlUtilization_t util{};
    nvmlDeviceGetUtilizationRates(handle, &util);

    nvmlMemory_t memInfo{};
    nvmlDeviceGetMemoryInfo(handle, &memInfo);
    double total_mem = static_cast<double>(memInfo.total);
    double used_device_mem_MB = memInfo.used / 1024.0 / 1024.0;
    device_total_fbmem_[i] = used_device_mem_MB;

    unsigned int info_count = 128;
    std::vector<nvmlProcessInfo_t> proc_infos(info_count);
    nvmlReturn_t res =
        nvmlDeviceGetComputeRunningProcesses_v3(handle, &info_count, proc_infos.data());
    if (res == NVML_ERROR_INSUFFICIENT_SIZE) {
      proc_infos.resize(info_count);
      nvmlDeviceGetComputeRunningProcesses_v3(handle, &info_count, proc_infos.data());
    }

    for (unsigned int j = 0; j < info_count; ++j) {
      pid_t pid = proc_infos[j].pid;
      if (std::find(pids.begin(), pids.end(), pid) == pids.end()) continue;

      nvml_process_stats& stat = gpu_stats[pid];
      stat.sm_util =+ util.gpu*100; // SM% from device-level util
      stat.fb_mem_used =+ proc_infos[j].usedGpuMemory; // bytes
      stat.gpu_mem_used_pct =+
          (total_mem > 0.0) ? (proc_infos[j].usedGpuMemory * 100.0 / total_mem) : 0.0;
    }
  }
}



monitored_value_map const nvmlmon::get_text_stats() {
  monitored_value_map stats;
  for (const auto& [pid, s] : gpu_stats) {
    stats["pid_" + std::to_string(pid) + "_gpufbmem_MB"] = s.fb_mem_used / 1024.0 / 1024.0;
    stats["pid_" + std::to_string(pid) + "_gpumempct"] = s.gpu_mem_used_pct;
    stats["pid_" + std::to_string(pid) + "_gpusmpct"] = s.sm_util;
  }
  return stats;
}

monitored_value_map const nvmlmon::get_json_total_stats() {
  monitored_value_map totals;
  double total_fb_mem_MB = 0.0;
  double total_mem_pct = 0.0;
  double avg_sm = 0.0;

  for (const auto& [_, s] : gpu_stats) {
    total_fb_mem_MB += s.fb_mem_used / 1024.0 / 1024.0;
    total_mem_pct += s.gpu_mem_used_pct;
    avg_sm += s.sm_util;
  }

  size_t count = gpu_stats.size();
  if (count > 0) {
    total_mem_pct /= count;
    avg_sm /= count;
  }
 
  double device_total_fbmem_MB = 0.0;
  for (const auto& [gpu_idx, fb] : device_total_fbmem_)
    device_total_fbmem_MB += fb;

  totals["gpufbmem"] = total_fb_mem_MB;
  totals["gpumempct"] = total_mem_pct;
  totals["gpusmpct"] = avg_sm;
  totals["gpu_total_fbmem_MB"] = device_total_fbmem_MB;

  return totals;
}

monitored_average_map const nvmlmon::get_json_average_stats(
    unsigned long long /*elapsed_clock_ticks*/) {

}

parameter_list const nvmlmon::get_parameter_list() {
  return {
      {"gpufbmem", "MB"},
      {"gpumempct", "%"},
      {"gpusmpct", "%"},
      {"gpu_total_fbmem_MB", "MB"},
  };
}

void const nvmlmon::get_hardware_info(nlohmann::json& hw_json) {
  for (const auto& d : devices) {
    hw_json["gpus"].push_back({
        {"index", d.index},
        {"name", d.name},
        {"total_mem_MB", d.total_mem / 1024.0 / 1024.0}
    });
  }
}

void const nvmlmon::get_unit_info(nlohmann::json& unit_json) {
  unit_json = {
      {"gpufbmem", "MB"},
      {"gpumempct", "%"},
      {"gpusmpct", "%"},
      {"gpu_total_fbmem_MB", "MB"}
  };
}

