#!/usr/bin/env bash
# install-custom-kernel.sh
# Creates a clean GRUB entry for the kernel built in a given tree (default: ~/linux).

set -euo pipefail

# --- Config / Args ------------------------------------------------------------
KDIR="${1:-$HOME/linux}"   # kernel source dir (arg1 overrides)
CONSOLE_ARGS="${CONSOLE_ARGS:-console=tty0 console=ttyS0,115200n8 rootdelay=5}"  # tweak if you like

# --- Helpers ------------------------------------------------------------------
need() { command -v "$1" >/dev/null || { echo "Missing: $1" >&2; exit 1; }; }

need sudo
need grub-probe
need grub-mkconfig

if [[ ! -d "$KDIR" || ! -f "$KDIR/Makefile" ]]; then
  echo "Kernel tree not found at: $KDIR" >&2
  echo "Usage: $0 [path/to/kernel/tree]" >&2
  exit 1
fi

cd "$KDIR"

# --- Derive kernel release & ensure image exists ------------------------------
KREL=$(make -s kernelrelease)
echo "[i] Kernel release: $KREL"

if [[ ! -f arch/x86/boot/bzImage ]]; then
  echo "[i] bzImage not found — building it..."
  make -j"$(nproc)" bzImage
fi

# --- Install kernel image into /boot ------------------------------------------
echo "[i] Installing kernel image to /boot/vmlinuz-$KREL"
sudo cp arch/x86/boot/bzImage "/boot/vmlinuz-$KREL"

# --- Ensure initramfs exists (build if missing) -------------------------------
if [[ ! -f "/boot/initramfs-$KREL.img" ]]; then
  echo "[i] Building initramfs for $KREL (mkinitcpio)..."
  sudo mkinitcpio -k "$KREL" -g "/boot/initramfs-$KREL.img"
else
  echo "[i] Found existing /boot/initramfs-$KREL.img"
fi

# --- Probe UUIDs and Btrfs subvol (if any) ------------------------------------
ROOT_UUID=$(sudo grub-probe --target=fs_uuid /)
BOOT_UUID=$(sudo grub-probe --target=fs_uuid /boot 2>/dev/null || sudo grub-probe --target=fs_uuid /)
SUBVOL=$(findmnt -no OPTIONS / | sed -n 's/.*\(subvol=[^, ]*\).*/\1/p')

echo "[i] ROOT UUID: $ROOT_UUID"
echo "[i] BOOT UUID: $BOOT_UUID"
[[ -n "${SUBVOL:-}" ]] && echo "[i] Btrfs subvol detected: $SUBVOL"

# --- Restore safe 40_custom stub (so grub-mkconfig won’t error) --------------
echo "[i] Restoring /etc/grub.d/40_custom stub"
sudo bash -c 'cat > /etc/grub.d/40_custom << "EOF"
#!/bin/sh
exec tail -n +3 $0
# This file provides an easy way to add custom menu entries.
# Type your menu entries below this comment, as GRUB script, not shell.
EOF
chmod +x /etc/grub.d/40_custom'

# --- Write our custom GRUB entry (GRUB reads this directly at boot) -----------
echo "[i] Writing /boot/grub/custom.cfg"
sudo tee /boot/grub/custom.cfg >/dev/null <<EOF
menuentry 'Arch (custom ${KREL})' {
    search --no-floppy --fs-uuid --set=root ${BOOT_UUID}
    linux  /vmlinuz-${KREL} root=UUID=${ROOT_UUID} rw ${CONSOLE_ARGS} ${SUBVOL:+rootflags=${SUBVOL}} ${SUBVOL:+rootfstype=btrfs}
    initrd /initramfs-${KREL}.img
}
EOF

# --- Regenerate the main GRUB config (good hygiene) ---------------------------
echo "[i] Regenerating grub.cfg"
sudo grub-mkconfig -o /boot/grub/grub.cfg

echo
echo "[✓] Done. Reboot and select:  Arch (custom ${KREL})"
echo "[i] Tip: if you use serial only, run QEMU with:  -serial mon:stdio"

