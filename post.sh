#!/usr/bin/env bash
# Make .vtk.series indexes for slice outputs so ParaView can load all timesteps.

cd postProcessing/T_slice || { echo "Can't cd to postProcessing/T_slice"; exit 1; }

# Update this list to match your actual file basenames:
planes=(planeZ001 planeY0)

# Numeric sort of time directories like 0.0002, 1e-6
times=$(ls -1d [0-9]* 2>/dev/null | sort -g)

for plane in "${planes[@]}"; do
  out="${plane}.vtk.series"
  echo '{ "file-series-version": "1.0", "files": [' > "$out"

  first=1
  count=0
  for t in $times; do
    file="$t/${plane}.vtk"
    if [ -f "$file" ]; then
      [ $first -eq 1 ] || echo "," >> "$out"
      first=0
      count=$((count+1))
      printf '  { "name": "%s", "time": %s }' "$file" "$t" >> "$out"
    fi
  done

  echo >> "$out"
  echo "] }" >> "$out"

  if [ $count -eq 0 ]; then
    rm -f "$out"
    echo "Skipped $plane: no files found (check name)."
  else
    echo "Wrote $out with $count steps."
  fi
done

