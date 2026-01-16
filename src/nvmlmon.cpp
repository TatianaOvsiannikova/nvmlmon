#include "nvmlmon.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

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

  devices.clear();
  devices.reserve(ngpus);

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

  // per-GPU rolling timestamp for nvmlDeviceGetProcessUtilization
  last_ts_.assign(ngpus, 0ULL);

  valid = true;
}

void nvmlmon::shutdown_nvml() {
  if (valid) nvmlShutdown();
}

bool debug_enabled = false;
inline void log_debug(const std::string& msg) {
  if (!debug_enabled) return;
  std::cerr << "[nvmlmon] " << msg << '\n';
}
void nvmlmon::update_stats(const std::vector<pid_t>& pids, const std::string read_path) {
  if (!valid) return;

  auto nvml_err = [](nvmlReturn_t r) {
    return std::string(nvmlErrorString(r));
  };

  gpu_stats.clear();
  device_total_fbmem_.clear();

  if (!read_path.empty()) {
    log_debug("read_path is non-empty ('" + read_path + "'), returning early (no NVML collection).");
    return;
  }

  // Fast PID membership test
  std::unordered_set<pid_t> watched(pids.begin(), pids.end());
  log_debug("update_stats: watched PIDs count=" + watched.size());

  std::unordered_map<unsigned int, bool> activegpus;

  for (unsigned int i = 0; i < ngpus; ++i) {
    nvmlDevice_t handle;
    nvmlReturn_t hr = nvmlDeviceGetHandleByIndex_v2(i, &handle);
    if (hr != NVML_SUCCESS) {
       log_debug("GPU " + std::to_string(i) +": nvmlDeviceGetHandleByIndex_v2 failed: " + nvml_err(hr));
      continue;
    }

    char name[96] = {0};
    nvmlReturn_t nr = nvmlDeviceGetName(handle, name, sizeof(name));
    log_debug("GPU " + std::to_string(i) +": handle OK, name=" + (nr == NVML_SUCCESS ? name : "<unknown>"));

    // -------------------------
    // PROCESS MEMORY (compute + graphics)
    // -------------------------
    auto collect_proc_mem = [&](const char* tag, auto fn_get_procs) {
      unsigned int count = 0;

      nvmlReturn_t r1 = fn_get_procs(handle, &count, nullptr);
      if (r1 != NVML_ERROR_INSUFFICIENT_SIZE && r1 != NVML_SUCCESS) {
        log_debug("GPU " + std::to_string(i) +" [" + tag +"]: first call failed: " + nvml_err(r1));
        return;
      }
      if (count == 0) {
        log_debug("GPU " + std::to_string(i) +" [" + tag + "]: no running processes reported.");

        return;
      }

      std::vector<nvmlProcessInfo_t> procInfos(count);
      nvmlReturn_t r2 = fn_get_procs(handle, &count, procInfos.data());
      if (r2 != NVML_SUCCESS) {
        log_debug("GPU " + std::to_string(i) +" [" + tag +"]: second call failed: " + nvml_err(r2));
        return;
      }

      log_debug("GPU " + std::to_string(i) +" [" + tag +"]: process entries returned=" +std::to_string( count));

      for (unsigned int j = 0; j < count; ++j) {
        pid_t pid = procInfos[j].pid;
        unsigned long long memB = procInfos[j].usedGpuMemory;

        // Print all, even non-watched, to understand what NVML sees
        log_debug("GPU " + std::to_string(i) +" [" + tag + "]: pid=" + std::to_string(pid) + " usedGpuMemory(B)=" + std::to_string( memB) + (watched.count(pid) ? " [WATCHED]" : ""));


        if (!watched.count(pid)) continue;

        activegpus[i] = true;

        nvml_process_stats& s = gpu_stats[pid];
        s.fb_mem_used += memB;  // bytes
      }
    };

    collect_proc_mem("compute",
      [&](nvmlDevice_t h, unsigned int* c, nvmlProcessInfo_t* p) {
        return nvmlDeviceGetComputeRunningProcesses_v3(h, c, p);
      }
    );
    collect_proc_mem("graphics",
      [&](nvmlDevice_t h, unsigned int* c, nvmlProcessInfo_t* p) {
        return nvmlDeviceGetGraphicsRunningProcesses_v3(h, c, p);
      }
    );

    // -------------------------
    // PROCESS UTILIZATION (SM/MEM)
    // -------------------------
    struct Latest {
      unsigned long long ts = 0;
      unsigned int sm = 0;
      unsigned int mem = 0;
    };
    std::unordered_map<pid_t, Latest> latest_by_pid;

    log_debug("GPU " + std::to_string(i) + ": last_ts_ before=" + std::to_string(last_ts_[i]));

    unsigned int utilCount = 0;
    nvmlReturn_t ur1 = nvmlDeviceGetProcessUtilization(handle, nullptr, &utilCount, last_ts_[i]);

    if (ur1 != NVML_ERROR_INSUFFICIENT_SIZE && ur1 != NVML_SUCCESS) {
      log_debug("GPU " + std::to_string(i) +  ": GetProcessUtilization(first) failed: " + nvml_err(ur1));
    } else if (utilCount == 0) {
      log_debug("GPU " + std::to_string(i) +   ": GetProcessUtilization: utilCount=0 (no samples since last_ts_)");

    } else {
      log_debug("GPU " + std::to_string(i) + ": GetProcessUtilization: utilCount=" + std::to_string(utilCount));

      std::vector<nvmlProcessUtilizationSample_t> samples(utilCount);
      nvmlReturn_t ur2 = nvmlDeviceGetProcessUtilization(handle, samples.data(), &utilCount, last_ts_[i]);
      if (ur2 != NVML_SUCCESS) {
        log_debug("GPU " + std::to_string(i) + ": GetProcessUtilization(second) failed: " + nvml_err(ur2));
      } else {
        for (unsigned int k = 0; k < utilCount; ++k) {
          const auto& samp = samples[k];
          log_debug("GPU " + std::to_string(i) + ": sample k=" + std::to_string(k)
                    + " pid=" + std::to_string(samp.pid)
                    + " ts=" + std::to_string(samp.timeStamp)
                    + " smUtil=" + std::to_string(samp.smUtil)
                    + " memUtil=" + std::to_string(samp.memUtil)
                    + (watched.count(samp.pid) ? " [WATCHED]" : ""));

          if (samp.timeStamp > last_ts_[i]) last_ts_[i] = samp.timeStamp;

          if (!watched.count(samp.pid)) continue;

          activegpus[i] = true;

          auto& L = latest_by_pid[samp.pid];
          if (samp.timeStamp >= L.ts) {
            L.ts  = samp.timeStamp;
            L.sm  = samp.smUtil;
            L.mem = samp.memUtil;
          }
        }
      }
    }

    log_debug("GPU " + std::to_string(i) +": last_ts_ after=" + std::to_string(last_ts_[i]));
    // Apply latest samples to gpu_stats
    for (const auto& [pid, L] : latest_by_pid) {
      auto& s = gpu_stats[pid];
      s.sm_util  = L.sm;
      s.mem_util = L.mem;
      log_debug("GPU " + std::to_string(i) + ": APPLY pid=" + std::to_string(pid)
                + " latest_ts=" + std::to_string(L.ts)
                + " sm=" + std::to_string(L.sm)
                + " mem=" + std::to_string(L.mem));
    }
  }

  // Final summary (per watched PID)
  log_debug("FINAL SUMMARY (watched PIDs that appeared):");
  for (const auto& [pid, s] : gpu_stats) {
    log_debug("pid=" + std::to_string(pid)
              + " fb_mem_used(B)=" + std::to_string(s.fb_mem_used)
              + " sm_util(%)=" + std::to_string(s.sm_util)
              + " mem_util(%)=" + std::to_string(s.mem_util));
  }
}


monitored_value_map const nvmlmon::get_text_stats() {
  monitored_value_map stats;

  // prmon sums per-process values for watched PIDs; it does NOT average
  double fbmem_MB = 0.0;
  double mempct_sum = 0.0;
  double smpct_sum  = 0.0;

  for (const auto& [pid, s] : gpu_stats) {
    fbmem_MB   += s.fb_mem_used ; // bytes
    mempct_sum += s.mem_util;                     
    smpct_sum  += s.sm_util;                      
  }

  stats["gpufbmem"]  = fbmem_MB;
  stats["gpumempct"] = mempct_sum;
  stats["gpusmpct"]  = smpct_sum;

  return stats;
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


// Collect related hardware information using NVML (no nvidia-smi)
void const nvmlmon::get_hardware_info(nlohmann::json& hw_json) {
  // Record the number of GPUs
  hw_json["HW"]["gpu"]["nGPU"] = ngpus;

  for (unsigned int i = 0; i < ngpus; ++i) {
    nvmlDevice_t handle;
    nvmlReturn_t r = nvmlDeviceGetHandleByIndex_v2(i, &handle);
    if (r != NVML_SUCCESS) {
      continue;
    }

    std::string gpu_number = "gpu_" + std::to_string(i);

    // GPU name
    char name_buf[96] = {0};
    r = nvmlDeviceGetName(handle, name_buf, sizeof(name_buf));
    hw_json["HW"]["gpu"][gpu_number]["name"] =
        (r == NVML_SUCCESS) ? std::string(name_buf) : std::string("unknown");

    // Max SM clock (MHz): NVML equivalent of clocks.max.sm
    unsigned int sm_freq_mhz = 0;
    r = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM, &sm_freq_mhz);
    if (r == NVML_SUCCESS) {
      hw_json["HW"]["gpu"][gpu_number]["sm_freq"] = sm_freq_mhz;  // MHz, like nvidia-smi nounits
    } else {
      // Fallback: current SM clock if max isn't available
      unsigned int cur_sm_mhz = 0;
      if (nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM, &cur_sm_mhz) == NVML_SUCCESS) {
        hw_json["HW"]["gpu"][gpu_number]["sm_freq"] = cur_sm_mhz;
      }
    }

    // Total memory: NVML returns bytes
    nvmlMemory_t memInfo{};
    r = nvmlDeviceGetMemoryInfo(handle, &memInfo);
    if (r == NVML_SUCCESS) {

      hw_json["HW"]["gpu"][gpu_number]["total_mem"] =
          static_cast<unsigned long long>(memInfo.total );
    }
  }
}
