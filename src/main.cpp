#include "nvmlmon.h"

#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
bool running = true;
void sigint_handler(int) { running = false; }
} 

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: nvmlmon [--interval N] [--output file] -- <command> [args...]\n";
    return 1;
  }

  double interval = 1.0;
  std::string output_file;
  int cmd_index = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--interval" && i + 1 < argc) {
      interval = std::stod(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
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

  std::vector<char*> cmd_args;
  for (int i = cmd_index; i < argc; ++i) cmd_args.push_back(argv[i]);
  cmd_args.push_back(nullptr);

  pid_t child_pid = fork();
  if (child_pid == 0) {
    execvp(cmd_args[0], cmd_args.data());
    perror("execvp failed");
    _exit(127);
  }

  std::signal(SIGINT, sigint_handler);
  std::signal(SIGTERM, sigint_handler);

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

  std::cout << "Monitoring PID " << child_pid << " every " << interval << " s\n";

  while (running) {
    int status;
    pid_t result = waitpid(child_pid, &status, WNOHANG);
    if (result == child_pid) {
      std::cout << "Process exited.\n";
      break;
    }

    monitor.update_stats({child_pid});
    auto stats = monitor.get_text_stats();

    const auto timestamp_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();

    std::ostringstream line;
    line << "timestamp=" << timestamp_ms;
    for (const auto& [key, value] : stats) {
      line << " " << key << "=" << value;
    }
    line << "\n";
 
    if (fout.is_open())
      fout << line.str();
    else
      std::cout << line.str();

    std::this_thread::sleep_for(std::chrono::duration<double>(interval));
  }

  if (fout.is_open()) fout.close();
  return 0;
}

