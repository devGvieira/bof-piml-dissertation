#!/bin/bash
cd "/home/icarus/Master's Dissertation"
grep -oP '\\cite\{[^}]+\}' Capitulos/*.tex | \
  sed 's/.*\\cite{//' | sed 's/}//' | tr ',' '\n' | \
  sed 's/^ *//' | sort -u
