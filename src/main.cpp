#include "nvmlmon.h"

#include <chrono>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_set>

#include <sys/wait.h>
#include <unistd.h>

namespace nvmlm{
bool running = true;
void sig_handler(int) { running = false; }
} 

// -------- prmon-like child PID collection (/proc/<pid>/task/<pid>/children) -----

static std::vector<pid_t> read_children_proc(pid_t pid) {
  std::vector<pid_t> children;
  std::ostringstream path;
  path << "/proc/" << pid << "/task/" << pid << "/children";

  std::ifstream in(path.str());
  if (!in.is_open()) return children;

  pid_t cpid;
  while (in >> cpid) {
    if (cpid > 0) children.push_back(cpid);
  }
  return children;
}

static bool has_proc_children(pid_t pid) {
  std::ostringstream path;
  path << "/proc/" << pid << "/task/" << pid << "/children";
  std::ifstream in(path.str());
  return in.is_open();
}

static std::vector<pid_t> collect_childrens_proc(pid_t root_pid) {
  std::vector<pid_t> result;
  std::unordered_set<pid_t> seen;
  std::vector<pid_t> stack;

  stack.push_back(root_pid);
  seen.insert(root_pid);

  while (!stack.empty()) {
    pid_t pid = stack.back();
    stack.pop_back();

    result.push_back(pid);

    for (pid_t c : read_children_proc(pid)) {
      if (c <= 0) continue;
      if (seen.insert(c).second) stack.push_back(c);
    }
  }
  return result;
}

// ------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: nvmlmon [-i N] [-o file] -- <command> [args...]\n";
    return 1;
  }

  double interval_arg = 1.0;
  std::string output_file;
  int cmd_index = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-i" && i + 1 < argc) {
      interval_arg = std::stod(argv[++i]);
    } else if (arg == "-o" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "--") {
      cmd_index = i + 1;
      break;
    }
  }

  if (cmd_index == -1 || cmd_index >= argc) {
    std::cerr << "Error: no command provided after '--'\n";
    return 1;
  }

  // prmon uses integer seconds logic (time(0)); keep that here
  const time_t interval =
      (interval_arg < 0.0) ? 0 : static_cast<time_t>(interval_arg);

  std::vector<char*> cmd_args;
  for (int i = cmd_index; i < argc; ++i) cmd_args.push_back(argv[i]);
  cmd_args.push_back(nullptr);

  pid_t m_pid = fork();
  if (m_pid == 0) {
    execvp(cmd_args[0], cmd_args.data());
    perror("execvp failed");
    _exit(127);
  }

  std::signal(SIGINT, nvmlm::sig_handler);
  std::signal(SIGTERM, nvmlm::sig_handler);

  nvmlmon monitor;
  if (!monitor.is_valid()) {
    std::cerr << "NVML not available — exiting.\n";
    return 1;
  }

  std::ofstream fout;
  if (!output_file.empty()) {
    fout.open(output_file);
    if (!fout.is_open()) {
      std::cerr << "Error: cannot open output file '" << output_file << "'\n";
      return 1;
    }
  }

  std::cout << "Montoring PID tree rooted at " << m_pid  << " every "
            << interval << " s \n";


  const std::vector<std::string> columns = {
      "gpufbmem", "gpumempct", "gpusmpct"
  };

  bool printed_header = false;
  const time_t start_ts = time(0);

  // Cache kernel capability check 
  const bool modern_kernel = has_proc_children(m_pid);
  if (!modern_kernel) {
    std::cerr << "Warning: /proc/<pid>/task/<pid>/children not readable; "
                 "monitoring only the mother PID.\n";
  }

  time_t lastIteration = time(0) - interval;

  while (nvmlm::running) {
    int status = 0;
    pid_t result = waitpid(m_pid, &status, WNOHANG);
    if (result == m_pid) {
      std::cout << "Process exited.\n";
      break;
    }

    if (time(0) - lastIteration > interval) {
      lastIteration = time(0);

      // PID + all childrens 
      std::vector<pid_t> pids_to_monitor;
      if (modern_kernel) {
        pids_to_monitor = collect_childrens_proc(m_pid);
      } else {
        pids_to_monitor = {m_pid};
      }

      monitor.update_stats(pids_to_monitor);
      auto stats = monitor.get_text_stats();

      const time_t now = time(0);
      const time_t wtime = now - start_ts;

      // Header once 
      if (!printed_header) {
        std::ostream& out =
            fout.is_open() ? static_cast<std::ostream&>(fout)
                           : static_cast<std::ostream&>(std::cout);
        out << "Time\twtime";
        for (const auto& c : columns) out << "\t" << c;
        out << "\n";
        printed_header = true;
      }

      std::ostream& out =
          fout.is_open() ? static_cast<std::ostream&>(fout)
                         : static_cast<std::ostream&>(std::cout);
      out << now << "\t" << wtime;

      for (const auto& c : columns) {
        auto it = stats.find(c);
        // prmon prints 0 if field missing; match that
        const double v = (it != stats.end()) ? static_cast<double>(it->second)
                                             : 0.0;
        out << "\t" << ((c == "gpufbmem") ? v / 1024 : v);
      }
      out << "\n";

      if (fout.is_open()) fout.flush();
    }


    // prmon sleeps 200ms between checks
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  if (fout.is_open()) fout.close();
  return 0;
}
