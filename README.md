# Arch Linux Kernel with Adaptive RL Scheduler

This repository contains a custom Arch Linux kernel fork that integrates an adaptive reinforcement learning (RL)-based CPU scheduler into the Linux scheduling subsystem.

---

## Overview


This kernel introduces a lightweight RL policy that:
- Observes runtime scheduling state
- Makes per-decision scheduling adjustments
- Adapts behavior based on observed workload patterns

All decisions are made entirely in-kernel with deterministic fixed-point inference.

---

## Key Features

- Reinforcement learning-based scheduling policy (PPO-trained)
- Fully in-kernel neural network inference
- Q32.32 fixed-point arithmetic (no floating point)
- Compatible with CFS / EEVDF scheduling paths
- Constant-time inference with low overhead

---

## Scheduler Design

The RL agent observes a compact state representation including recent wait time, burst time, and scheduling context.  
Actions influence scheduling parameters such as task slice length and effective priority.

The reward function balances:
- Wait time
- CPU burst efficiency
- Context switch overhead
- Fairness

---

## Building
### Dependencies

The following tools are required to build the kernel:
- Compiler toolchain (gcc/clang, make)
- bc
- cpio
- Python 3
- pahole (from `dwarves`)
- git

#### Arch Linux
```bash
sudo pacman -S base-devel bc cpio python pahole git
```

#### Debian / Ubuntu
```bash
sudo apt install build-essential bc cpio python3 dwarves git
```

#### Fedora
```bash
sudo dnf install @development-tools bc cpio python3 dwarves git
```

Other distributions should install equivalent packages using their system package manager.

### Build and Install
```bash
git clone https://github.com/vrushang1234/arch-kernel.git
cd arch-kernel
make -j$(nproc)
sudo make modules_install
sudo make install
```

Reboot into the new kernel after updating your bootloader.


