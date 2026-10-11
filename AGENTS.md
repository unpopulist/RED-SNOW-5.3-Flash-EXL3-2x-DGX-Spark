# AGENTS.md: set up RED-SNOW-5.3-FLASH on two DGX Sparks

These instructions are for an AI coding agent working on behalf of a human operator. Follow the steps in order.
After each step, run its check and stop if the check fails. Every command can be copied as is, except the
`<placeholders>`, which you fill from the operator's answers or from your preflight output.

Terms used below:

- **head**: the Spark that runs rank 0 and serves the API. You work on the head.
- **worker**: the Spark that runs rank 1.
- **link**: the QSFP cable between them, carrying RoCE.

## Before you start: rules

1. **Never delete or overwrite model folders, Hugging Face caches or other people's files.** Write new model data
   only into the folders named in this document, or into folders the operator gives you. Keep the rank-1 folder in
   its own directory. Never copy it into a full pack's folder: it has its own `config.json` and index files.
2. **Ask before you stop anything.** If port 8000 is taken, or a GPU process or container is running, report what
   it is and wait for the operator's decision. Never stop, kill or remove services, containers or processes you
   did not start.
3. **Ask before you change the system.** That includes drivers, kernel modules, network configuration, sysctl,
   `/etc`, packages that need `sudo`, and Docker daemon settings.
4. **Licenses and use.**
   - The DFlash2 drafter is CC BY-NC-ND 4.0 (non-commercial only). Before you download it, ask the operator to
     confirm that the deployment is non-commercial. If it is not, use `DRAFTER=none`.
   - RED-SNOW's model card limits use to experienced operators, in environments they own or are authorized to
     assess, with a human in control of scope, targets, credentials, tool execution and destructive actions. Show
     the operator this notice, with the link https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16.
5. Never put secrets (Hugging Face tokens, passwords) into files in this repository or into command lines that
   get logged.

## Step 0: collect the facts

Ask the operator for these, or find them:

| Placeholder | What | Example form |
|---|---|---|
| engine | `unpopulist/TensorSnow`, tag `redsnow-v1.0.0` | the engine fork (TensorFold + RED-SNOW) |
| `<engine-dir>` | Where to clone the engine on the head (an absolute path the operator owns) | `/opt/redsnow-engine` |
| `<worker-ssh>` | How the head reaches the worker by ssh | `user@worker-host` |
| `<rank0-roce-ip>`, `<rank1-roce-ip>` | Each node's IPv4 address on the link | found in step 1 |
| `<rank0-host>` | Where clients reach the head's API | hostname or LAN address |

## Step 1: preflight (both nodes)

Run each block on the head. Then run it on the worker through `ssh <worker-ssh> '<command>'`.

```bash
# GPU and driver: one GB10, driver loaded, nothing else using it
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# memory: MemAvailable should be ~115 GiB or more (the launcher refuses below 110 GiB)
awk '/MemAvailable/ {printf "%.1f GiB available\n", $2/1048576}' /proc/meminfo
# disk: ~200 GB free for models (default /var/tmp/models) and ~45 GB under Docker's root
df -h /var/tmp "$(docker info -f '{{.DockerRootDir}}')"
# docker with the NVIDIA runtime and buildx; RDMA device present
docker info 2>/dev/null | grep -i -m1 nvidia && docker buildx version && ls /dev/infiniband/uverbs0
# the link: RDMA device to netdev map, and the IPv4 address on the netdev that is "Up"
ibdev2netdev 2>/dev/null || ls /sys/class/infiniband
ip -br -4 addr show enp1s0f0np0
```

On the head only:

```bash
ssh -o BatchMode=yes <worker-ssh> true && echo "ssh ok"        # must not ask for a password
[ "$(ssh <worker-ssh> 'echo $HOME')" = "$HOME" ] && echo "same HOME"   # else set SHIM_DIR in .env
ping -c3 -I enp1s0f0np0 <rank1-roce-ip>                         # the worker over the link
ss -ltnp 'sport = :8000'; ss -ltnp 'sport = :29551'             # both must print nothing
```

Pass criteria:

- Both nodes: one GB10 with no other compute process, ≥110 GiB available (≈115 preferred), and enough disk.
- Docker has the NVIDIA runtime and `uverbs0` exists.
- Each node has an IPv4 address on the link netdev, and the ping answers.
- Passwordless ssh works and both nodes have the same `$HOME`.
- Ports 8000 and 29551 are free.

If the netdev or RDMA device names differ from `enp1s0f0np0` / `rocep1s0f0`, note the real names for `IFACE` and
`HCA` in step 3. If a port is taken or the GPU is busy, report it and ask (rule 2).

## Step 2: clone the engine fork (head)

```bash
git clone --branch redsnow-v1.0.0 https://github.com/unpopulist/TensorSnow <engine-dir>
git -C <engine-dir> log -1 --oneline
```

The worker needs no checkout. It receives the image and a small launcher shim.

## Step 3: settings

```bash
cd <recipe-dir>                       # this repository
cp .env.example .env
```

Edit `.env` and set `ENGINE_DIR=<engine-dir>`, `HEAD_ADDR=<rank0-roce-ip>`, `WORKER_ADDR=<rank1-roce-ip>` and
`WORKER_SSH=<worker-ssh>`. Uncomment and set `IFACE` / `HCA` if step 1 showed other names. Leave `HEAD_GID` /
`WORKER_GID` empty: the launcher finds each node's RoCE v2 GID itself. Check that no placeholder is left:

```bash
grep -n '<' .env && echo "placeholders left: fix them" || echo "settings ok"
```

## Step 4: build the image and copy it to the worker (head)

```bash
./run.sh build
```

This takes a while the first time: it pulls the NVIDIA PyTorch base and compiles the CUDA extensions. It prints
`image: redsnow-tensorfold:<commit12>-<recipe8>` and `the worker has ...`. Check:

```bash
docker images redsnow-tensorfold
ssh <worker-ssh> docker images redsnow-tensorfold      # the same tag on both
```

## Step 5: model and drafter on both nodes

On the head (after the operator has confirmed the DFlash2 license, rule 4):

```bash
mkdir -p /var/tmp/models
hf download vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw --local-dir /var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw
hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir /var/tmp/models/GLM-5.3-Flash-DFlash2
```

Next, write rank 1's half of the weights on the head. This runs on the CPU only, writes ~93 GiB and can be
restarted. Use the image tag from step 4:

```bash
mkdir -p /var/tmp/models/redsnow-rank1
docker run --rm --user "$(id -u):$(id -g)" -v /var/tmp/models:/var/tmp/models --entrypoint python \
  redsnow-tensorfold:<image-tag> -m tensorfold.families.glm5_next.cuda.split \
  /var/tmp/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw --rank 1 /var/tmp/models/redsnow-rank1
ls /var/tmp/models/redsnow-rank1/*.rank1.safetensors | wc -l      # 36
```

Copy the rank-1 folder and the drafter to the worker. Use the same paths, and give the rank-1 folder its own
directory:

```bash
ssh <worker-ssh> mkdir -p /var/tmp/models
rsync -a --info=progress2 /var/tmp/models/redsnow-rank1/ <worker-ssh>:/var/tmp/models/redsnow-rank1/
rsync -a --info=progress2 /var/tmp/models/GLM-5.3-Flash-DFlash2/ <worker-ssh>:/var/tmp/models/GLM-5.3-Flash-DFlash2/
ssh <worker-ssh> 'ls /var/tmp/models/redsnow-rank1/*.rank1.safetensors | wc -l; du -sh /var/tmp/models/redsnow-rank1'
```

Optional, and only if the operator agrees: once the worker's copy checks out (36 files, ~94 GiB), the head's
`/var/tmp/models/redsnow-rank1` can be removed to free ~100 GB. Rank 0 does not use it. Never remove the full pack.

If you used other paths, set `MODEL_RANK0`, `MODEL_RANK1` and `DRAFTER` in `.env`.

## Step 6: start

```bash
./run.sh check                                  # every launch check; prints both rank commands; starts nothing
tmux new -d -s redsnow './run.sh start'         # serves in the foreground inside tmux
tmux capture-pane -pt redsnow | tail -20        # follow progress
```

Startup takes about 2.5 minutes. It is ready when the output shows `smoke test OK` and
`RED-SNOW-5.3-FLASH is serving on http://0.0.0.0:8000/v1`. If either rank dies, the launcher stops both. Logs:
`/tmp/tensorfold-redsnow-rank0.log` on the head and `/tmp/tensorfold-redsnow-rank1.log` on the worker.

To stop: `./run.sh stop`. It stops only the pair this launcher started.

## Step 7: smoke test

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "RED-SNOW-5.3-FLASH", "max_tokens": 32, "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "Reply with OK."}]}'
```

## Step 8: validation

**Needle in a ~60K-token prompt.** The answer must contain the passphrase. Prefill takes about a minute.

```bash
python3 - <<'PY'
import json, time, urllib.request
filler = [f"Line {i}: the river kept flowing past the old mill while the town slept." for i in range(4000)]
filler.insert(2000, "Remember this: the secret passphrase is PURPLE-HARBOR-42.")
body = {"model": "RED-SNOW-5.3-FLASH", "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "\n".join(filler) + "\n\nWhat is the secret passphrase? Answer with it only."}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t
text = r["choices"][0]["message"]["content"]; u = r["usage"]
print(f"{u['prompt_tokens']} prompt tokens in {dt:.0f} s ({u['prompt_tokens']/dt:.0f} tok/s incl. reply): {text!r}")
print("NEEDLE OK" if "PURPLE-HARBOR-42" in text else "NEEDLE FAILED")
PY
```

**Quick speed check, one request.** Expect roughly 30–40 tok/s on prose, and more on code or JSON.

```bash
python3 - <<'PY'
import json, time, urllib.request
body = {"model": "RED-SNOW-5.3-FLASH", "max_tokens": 512, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Write a 400-word essay about lighthouses."}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t
n = r["usage"]["completion_tokens"]; print(f"{n} tokens in {dt:.1f} s = {n/dt:.1f} tok/s")
PY
```

While these run, check memory on both nodes. The lowest free memory should stay at several GiB per node:

```bash
awk '/MemAvailable/ {printf "head %.1f GiB\n", $2/1048576}' /proc/meminfo
ssh <worker-ssh> "awk '/MemAvailable/ {printf \"worker %.1f GiB\n\", \$2/1048576}' /proc/meminfo"
```

Report all of the following to the operator:

- the needle result;
- the prompt tokens per second and the decode tokens per second;
- the lowest free memory you saw on each node.

## Troubleshooting

- **The launcher waits on "waiting for free memory to settle", or the engine refuses the window at start.** At 320K
  with a 14.5 GiB reserve the margin is a few tenths of a GiB. Free memory dips for a while after a big image load
  or file copy. Wait one or two minutes and run `./run.sh start` again. If it keeps failing, lower `CONTEXT` in
  `.env` (for example `262144`) and tell the operator. Do not lower `MEMORY_RESERVE_GIB`: the server needs that
  headroom under load. Do not drop caches or change system settings without asking.
- **"MemAvailable ... under 110 GiB".** Something else is using memory (another model server, a container, a large
  page cache from a copy). Report what it is (`ps aux --sort=-rss | head`, `docker ps`) and ask (rule 2).
- **"port 8000 is in use".** Find the owner (`ss -ltnp 'sport = :8000'`, `docker ps`) and ask the operator. Another
  port is possible with `PORT=8001` in `.env`.
- **GID, ping or RoCE errors** ("has no RoCE v2 GID", "does not answer over", or no "RoCE all-gather is live" line
  in the rank 0 log):
  - Check that `HEAD_ADDR` / `WORKER_ADDR` are the link addresses, not LAN ones.
  - Check `ibdev2netdev` shows the netdev "Up" on both nodes.
  - Check that `IFACE` / `HCA` match.
  - Check that the cable is seated.
  - To isolate the problem, try `COMM=nccl` in `.env`. It gives the same results, slightly slower.
- **"cannot ssh to ... without a password".** Set up a key from the head to the worker (ask the operator for the
  account to use).
- **"image ... missing on the worker" or "differs".** Run `./run.sh build` again.
- **A rank exits during loading.** Read the last lines of both logs (paths in step 6). Common causes are an
  incomplete rank-1 folder (re-run the split, then rsync), a drafter missing on one node, or another GPU process.
- **"a redsnow-rank* container is running".** A previous pair is still up. If you started it, run `./run.sh stop`.
  If not, ask.

## Done

When steps 6–8 pass, give the operator:

- the API URL: `http://<rank0-host>:8000/v1`, model name `RED-SNOW-5.3-FLASH`;
- how to stop the server: `./run.sh stop`;
- the validation numbers.
