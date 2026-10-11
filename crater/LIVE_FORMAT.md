# LIVE data for the dashboard

Start the system:  `python run_live.py`   (add `--link clean|default|harsh`)
It rewrites 3 files in the `live/` folder about twice per second. The dashboard only needs to READ them.
If a read fails for an instant (file being replaced), ignore it and read again at the next refresh.

## live/snapshot.json  - the map
{ "time_s": 42.1,
  "hazards": [ { "id": 1, "cls": "rock" | "crater",
                 "x": metres East of landing site, "y": metres North, "z": elevation (m), "r": radius (m),
                 "sig_major": 1-sigma ellipse major axis (m), "sig_minor": minor axis (m), "angle": ellipse angle (rad),
                 "age": seconds since last seen (grows over time -> fade the marker),
                 "n_obs": how many sightings were merged, "trust": 0..1 (colour by this) } ] }

## live/status.json  - counters and the network panel
{ "time_s", "source", "link": "clean|default|harsh", "expected_loss": 0.22,
  "link_stats":   { "sent", "dropped", "corrupted", "bytes_sent" },
  "ground_stats": { "rx_packets", "crc_fail", "duplicates" },
  "frames", "fps", "last_ms" (edge time per frame), "last_detections",
  "recent_packets": [ { "n", "t", "bytes", "radio_state": "good|bad", "fate": "delivered|DROPPED|CORRUPTED", "delay_s" } ],
  "recent_frames":  [ { "frame", "image", "detections", "packets", "bytes", "edge_ms" } ] }

## live/latest_frame.png - the rover's newest image with detections drawn on it

## Ready-made helpers
`from dashboard.plots import map_figure`  ->  `map_figure(snap["hazards"], [], None, snap["time_s"])` returns a matplotlib
figure of the map (works directly on snapshot.json).   Full example: dashboard/live_example.py
## Sending images from the rover
Copy .png/.jpg files into the folder `rover_inbox/` (optional: `name.json` next to an image holds its camera pose).
