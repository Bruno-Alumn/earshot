<div align="center">

<img src="docs/media/logo.png" width="128" height="128" alt="Earshot app icon">

# Earshot

**Every voice, on its own line.**

Live transcripts of your voice calls, one person at a time, on your own computer.

<a href="https://github.com/Bruno-Alumn/earshot/releases/latest"><img src="https://img.shields.io/badge/Download_for_Mac_%26_PC-000A3B?style=for-the-badge&logo=apple&logoColor=white" alt="Download Earshot for Mac and PC" height="36"></a>

<a href="https://github.com/Bruno-Alumn/earshot/releases/latest"><img src="https://img.shields.io/github/v/release/Bruno-Alumn/earshot?style=flat-square&color=000A3B&label=release" alt="Latest release"></a>
<img src="https://img.shields.io/badge/macOS-14%2B-1f2937?style=flat-square&logo=apple&logoColor=white" alt="macOS 14 or later">
<img src="https://img.shields.io/badge/Windows-10_%7C_11-1f2937?style=flat-square&logo=windows&logoColor=white" alt="Windows 10 or 11">
<img src="https://img.shields.io/badge/license-All_rights_reserved-6b7280?style=flat-square" alt="All rights reserved">

[Features](#features) · [Install](#install) · [Using Earshot](#using-earshot) · [Troubleshooting](#troubleshooting) · [Privacy](#privacy) · [Contribute](#contribute)

</div>

---

Earshot opens your voice platform's web client in its own window and
transcribes each person in the call from their own audio stream. When people
talk over each other, every line still lands under the right name, in the
order it was said. The transcript, your call history and the stats sit in a
dashboard next to the call, and no audio ever leaves your machine.

<p align="center">
  <img src="docs/media/dashboard.png" width="860" alt="Earshot's dashboard during a call: the live transcript on the left with each line under its speaker's name and colour, and on the right who is in the call, talk time per person, activity over the last hour and resource use. The names and conversation are made-up demo data.">
</p>

## Features

**Every person, separately**

- Each participant is transcribed from their own audio stream, so there is no guessing who spoke.
- When two people talk at once, their lines interleave by the moment each sentence was said.
- Names come from the call itself; someone whose name can't be read shows as `user-1234`.

**A dashboard for the call**

- Live transcript with search, and export to Markdown, text, SRT subtitles, CSV or JSON.
- Talk time, activity over time, a 10-minute timeline, crosstalk and most-said words.
- Past calls grouped by server and channel, with search across all of them.
- Compact mode puts the transcript and every panel on one screen, no scrolling.
- Arrange it your way: drag panels between columns, size or hide them, and add new ones from plugins.
- Personality reads: how each person comes across, as Big Five scores and interpersonal signals (including a narcissistic, self-focused style). The first read comes once they have said 40 lines and a new one after every 400 more, across all their calls, and every read is kept (personas.json). With an Anthropic API key saved in Settings, Claude Sonnet 5.5 doesn't score people directly: it tags each of their lines against a fixed codebook (brags, put-downs, banter, help, plans and so on), two independent passes per line, and only tags both passes agree on count. Fixed rules then turn the tag rates into scores, with minimum evidence for each score and "not rated" when there isn't enough. Their lines, and what others said just before and after, are sent to Anthropic for this. The Study window charts each person's reads over time, compares people, and checks reads against Big Five test scores you enter (for example from the free BFI-2). Without a key, the small local model writes them and nothing leaves your computer. An impression of their style, not a diagnosis.

**Several accounts, several calls**

- Add more accounts, each with its own call window and sign-in.
- Give any account its own proxy (HTTP, HTTPS, SOCKS4 or SOCKS5, with or without a username and password), so several accounts can sign in from different networks at once. Webshare's proxy list lines work as they come.
- Listener accounts get a silent microphone and start with their sound off, and are still transcribed.
- **Transcribe a channel** sends the next free account to any channel you pick; you click Join.
- **Auto-join** (off by default) lets a free account join channels you flag as it finds them open and not full — favourites first — and move on after a few quiet minutes. An account left alone in any channel for a minute (nobody there but your own accounts) moves to the busiest open flagged channel that none of your other accounts is in. Flag channels with the blue and star icons under **Channels**. Discord's terms forbid automating accounts, so it is yours to turn on and judge.
- Share a clean compact view of the call into the call itself when you screen share.

**On your computer only**

- Two local speech models: Granite 470M transcribes live, and Granite 2B can rewrite each line a few seconds later for the best accuracy.
- Updates itself from new releases.

## Install

**On a Mac** (Apple Silicon, macOS 14 or later):

1. Install [uv](https://docs.astral.sh/uv/), [Node.js](https://nodejs.org) and Xcode's command-line tools (`xcode-select --install`).
2. Run:
   ```sh
   git clone https://github.com/Bruno-Alumn/earshot.git ~/earshot
   cd ~/earshot/server && uv sync
   ~/earshot/app/mac/build.sh
   ```
3. Open **Earshot** from `~/Applications`.

**On a PC** (Windows 10 or 11, 64-bit):

1. Run `git clone https://github.com/Bruno-Alumn/earshot.git %USERPROFILE%\Earshot`.
2. Double-click **install.cmd** in that folder. The first run takes a few minutes: it installs Python, the speech model, the app and Node.js if it's missing, and adds Start-menu and desktop shortcuts.
3. Start **Earshot** from the Start menu.

With an NVIDIA graphics card the PC uses it for the most accurate model; without one it uses a CPU model at about 50× real time. [README-WINDOWS.md](README-WINDOWS.md) has the details.

> **A note about signing:** releases are not signed by Apple or Microsoft. The Mac app is built on your own Mac by `build.sh`, so macOS opens it without a warning. Install with `git clone` as above rather than from the release zip, or the app can't update itself.

## Using Earshot

1. **Sign in, once.** Show the call window with <kbd>⌘</kbd><kbd>D</kbd> and sign in to your voice platform there. Your Mac asks for the microphone the first time.
2. **Join a voice channel.** Transcription starts on its own; the dashboard fills in as people talk.
3. **Leave it running.** Closing a window keeps transcribing. Quit with <kbd>⌘</kbd><kbd>Q</kbd>, or from the tray icon on a PC.

| Action | What it does |
|:--|:--|
| <kbd>⌘</kbd><kbd>D</kbd> | Shows the call window |
| <kbd>⌘</kbd><kbd>⇧</kbd><kbd>T</kbd> | Picks a channel for the next free account to join |
| <kbd>⌘</kbd><kbd>⇧</kbd><kbd>K</kbd> | Switches compact mode on and off |
| <kbd>⌘</kbd><kbd>⇧</kbd><kbd>M</kbd> | Turns the call's sound off for you, while it keeps being transcribed |
| <kbd>⌘</kbd><kbd>⇧</kbd><kbd>F</kbd> | Searches every call |
| Layout button in the header | Customize: drag, size and hide panels, and change the columns |
| <kbd>⌘</kbd><kbd>,</kbd> | Opens Settings: start at login, auto-join, sentence splitting, speech detection, language, personality reads, updates |
| **Read** next to a name in Talk time | Writes a personality read of that person |

On a PC, use <kbd>Ctrl</kbd> in place of <kbd>⌘</kbd>. Accounts and the speech model are also under the status buttons at the top of the dashboard.

**Proxies.** Open **Accounts**, then **Proxy…** next to an account to give
that account's Discord window its own proxy. It accepts the formats proxied
services hand out: `host:port`, `host:port:username:password` (a Webshare
proxy list line, for example), `user:password@host:port`, and URLs such as
`http://user:password@host:port`, `https://…`, `socks4://…` and `socks5://…`.
With Webshare, a line from your proxy list (Dashboard → Proxy → List, or the
`proxy-manager` skill) pastes in as it comes; the rotating endpoint
`p.webshare.io:80:username:password` works too. Other
accounts keep their own setting, or your system network. The proxy covers that
window's Discord connection: sign-in, pages and the gateway. Discord's voice
media is WebRTC and Chromium may still send it outside the proxy, so it is not
a guarantee about the address the call itself uses.

Transcripts are saved in `transcripts/`, one Markdown and one JSON Lines file per day. Recording laws differ by place, and in many of them everyone in the call has to agree; following them is up to you.

## Troubleshooting

<details>
<summary><strong>Nobody is being transcribed</strong></summary>

Check that you're not deafened in the call: deafening stops the audio from
arriving at all. Use **Mute call audio** (<kbd>⌘</kbd><kbd>⇧</kbd><kbd>M</kbd>)
to silence the call for yourself instead. If the status at the top says the
call window isn't connected, restart the server from **Server › Restart Server**.
</details>

<details>
<summary><strong>A listener account says "deafened, nothing is being heard"</strong></summary>

That account is deafened in the call. Open its window from the Accounts list
and undeafen it; it stays muted, because listener accounts have no microphone.
</details>

<details>
<summary><strong>Lines show up several seconds late</strong></summary>

The model is falling behind. If Granite 2B is the only model on, turn 470M
back on from the model button at the top of the dashboard: it is much faster.
**Server › Show Server Log** shows what the server is doing.
</details>

<details>
<summary><strong>"Node.js is needed" when installing on a PC</strong></summary>

`install.cmd` installs Node.js with winget when it can. If winget isn't
available, install Node.js from [nodejs.org](https://nodejs.org) and run
`install.cmd` again.
</details>

If none of these help, [open an issue](https://github.com/Bruno-Alumn/earshot/issues/new) with the end of `server.log`.

## Privacy

Earshot has no accounts, analytics or telemetry, and call audio never leaves
your computer. Its own network requests are:

- **Speech and language models**, downloaded from Hugging Face the first time each is used (about 1.6 GB for the default speech model, 2 GB for personality reads).
- **Update checks** against this repository with `git`, when the app starts and every 6 hours.
- **The voice platform itself**, in the call window, as in a browser.

Transcripts, learned names, call history and any proxy credentials you enter
stay in the app's folder and are never committed to git.

### Permissions

| Permission | Why it's needed |
|:--|:--|
| **Microphone** | Lets your main account talk in calls. Listener accounts never get it. |
| **Screen & System Audio Recording** <sub>Mac</sub> | Only for sharing a screen or another app's window into a call. Sharing Earshot's own views works without it. |

## Contribute

- **Found something broken?** [Open an issue](https://github.com/Bruno-Alumn/earshot/issues/new).
- **Want to help?** Fork the repository and open a pull request against `main`. Say what you changed and be honest about what you tested.

Much of Earshot was built with AI coding agents, then compiled, tested and exercised on real hardware before merging.

<details>
<summary><strong>How it works</strong></summary>
<br>

The voice platform's web client receives each participant as a separate audio
track. `app/desktop/page-hook.js` is added to the page before the client's own
code runs; it reads each track at 16 kHz, without touching what you hear, and
the app passes the audio to a local Python server. The server cuts each
person's audio into utterances, transcribes them, splits them at sentence ends
using the model's own word timing, and writes `transcripts/*.md` and `*.jsonl`.

| Part | Where |
|:--|:--|
| Desktop app (Electron): dashboard, call windows, accounts, share picker, tray | `app/desktop/` |
| Transcription server and dashboard | `server/` |
| Mac app build, Windows installer | `app/mac/build.sh`, `install.cmd` |
| Tests | `test/` |

The server picks the engine for your machine: MLX on Apple Silicon, CUDA with
an NVIDIA GPU, otherwise ONNX Runtime on the CPU.
</details>

<details>
<summary><strong>Plugins</strong></summary>
<br>

A plugin is a folder in `plugins/` with a `manifest.json` and a JavaScript
module. It can add panels, mark up transcript lines, add header buttons and
react to new lines as they arrive. Turn plugins on in **Settings › Plugins**.
[PLUGINS.md](PLUGINS.md) documents the API, and `plugins/example-questions`
is a working example. Plugins can read your transcripts, so only turn on
ones you trust.
</details>

<details>
<summary><strong>Speech models</strong></summary>
<br>

Pick them with the model button at the top of the dashboard. Either can be on
alone, and turning both on makes 2B rewrite each line a few seconds after
470M writes it. Word error rate is on 120 real meeting clips (AMI), measured
on the maintainer's machines.

| Model | Words wrong | Speed (Mac / PC with NVIDIA) |
|:--|:--|:--|
| **Granite Speech 5.0 470M** (default) | 5.5% | 249× / 82× real time |
| Granite Speech 5.0 2B | 5.4% | 13× / 11× real time, about 9 GB of memory |

On a PC without an NVIDIA graphics card Granite can't keep up, so Earshot uses
Parakeet TDT 0.6B v3 on the processor instead (7.4% words wrong, about 30×
real time). Personality reads use Claude Sonnet 5.5 with an API key, or else Liquid AI LFM3 1.7B, which needs Apple Silicon
or an NVIDIA GPU.
</details>

<details>
<summary><strong>Build from source and tests</strong></summary>
<br>

```sh
git clone https://github.com/Bruno-Alumn/earshot.git
cd earshot/server && uv sync
cd ../app/desktop && npm ci && npx electron .
```

- `node test/app-test.js <dir>`: the real app against a fake call with four speakers.
- `node test/multi-test.js <dir>`: two accounts in separate calls, two in the same call, and Transcribe a channel.
- `node test/autojoin-test.js` and `node test/autojoin-app-test.js <dir>`: auto-join picks the right open, non-full, flagged channel, and an account left alone moves on (unit, then the real app).
- `node test/proxy-app-test.js <dir>`: per-account proxies (a Webshare list line, SOCKS5, changing and removing it).
- `node test/proxy-test.js`: the proxy formats accepted per account (no app needed).
- `cd server && uv run python ../test/timing_test.py`: lines split at the right times.
- `cd server && uv run python ../test/server_e2e.py`: the server alone, with people talking over each other.

CI runs these on macOS and Windows for every push. Pushing a `vX.Y.Z` tag that
matches `VERSION` publishes a release, which the app then offers as an update.
</details>

---

**All rights reserved:** © 2026 Bruno-Alumn.

<sub>`server/langwords.txt.gz` (word frequencies used to keep transcripts in English) is derived from [wordfreq](https://github.com/rspeer/wordfreq) data by Robyn Speer and is shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).</sub>

<sub>More apps by Bruno-Alumn: [SolStream](https://github.com/Bruno-Alumn/SolStream) · [Tuckaway](https://github.com/Bruno-Alumn/Tuckaway) · [ThockBox](https://github.com/Bruno-Alumn/ThockBox)</sub>
