// src/poly/conversations.hpp - the conversations a server has read, kept so that a prompt which starts like one of
// them is not read again from its first token.
//
// The engine holds one sequence at a time.  Coming back to an earlier point of a sequence takes two things: the
// recurrent state at that point, which is a checkpoint, and the rows its tokens left in the attention caches.  The
// rows of the sequence in the engine are there already.  A conversation that makes room for another is parked:
// its rows are copied to host memory, up to its last checkpoint.  What is saved is written once and never changed,
// so conversations that start alike share it, and a parked conversation comes back bit for bit.
//
// Nothing here knows the model.  E is the engine:
//     void reset();
//     size_t state_bytes() const;  void save_state(uint8_t*);  void load_state(const uint8_t*, int64_t pos);
//     size_t row_bytes() const;    void save_rows(int64_t first, int64_t n, uint8_t*);
//     void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace strata::poly {

struct CacheOptions {
    int checkpoints = 6;       ///< kept per conversation; 0: a prompt is reread unless it goes on from the last token
    int64_t every = 2048;      ///< a checkpoint every so many prompt tokens, besides the one at a prompt's end and
                               ///< the one where its last turn begins (turn_at()); 0: none
    int64_t park_mib = 0;      ///< host memory for parked conversations; 0: none is parked
    int park_slots = 4;        ///< parked conversations at most
    int64_t park_min = 1024;   ///< a conversation with fewer tokens before its last checkpoint is not parked
    int64_t session_min_free_mib = 4096;  ///< a session SAVE must leave this much free on its disk
    std::function<bool(size_t)> admit;   ///< whether the host can spare this many bytes; unset: it can
    FILE* log = nullptr;       ///< where parking is reported; the server reads these lines
};

namespace detail {

/// FNV-1a: corruption and identity checks for session files, not authentication.
inline uint64_t session_fnv(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
constexpr uint64_t kSessionFnvSeed = 1469598103934665603ull;

#ifndef _WIN32
struct SessionOut {
    int fd = -1;
    std::string tmp;
    uint64_t hash = kSessionFnvSeed;
    uint64_t total = 0;

    ~SessionOut() { abort(); }

    bool open(const std::string& path, std::string& err) {
        tmp = path + ".tmp." + std::to_string((long long) ::getpid());
        fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) {
            err = std::string("open: ") + std::strerror(errno);
            return false;
        }
        return true;
    }
    bool put_raw(const void* p, size_t n, std::string& err) {
        const uint8_t* q = static_cast<const uint8_t*>(p);
        while (n > 0) {
            const ssize_t w = ::write(fd, q, n);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) {
                err = std::string("write: ") + std::strerror(errno);
                return false;
            }
            q += w;
            n -= (size_t) w;
            total += (uint64_t) w;
        }
        return true;
    }
    bool put(const void* p, size_t n, std::string& err) {
        hash = session_fnv(hash, p, n);
        return put_raw(p, n, err);
    }
    bool finish(const std::string& path, std::string& err, uint64_t& bytes) {
        const uint64_t h = hash;
        static const char end[8] = {'P', 'O', 'L', 'Y', 'E', 'N', 'D', '!'};
        if (!put_raw(&h, sizeof h, err) || !put_raw(end, sizeof end, err)) return false;
        if (::fsync(fd) != 0) {
            err = std::string("fsync: ") + std::strerror(errno);
            return false;
        }
        if (::close(fd) != 0) {
            fd = -1;
            err = std::string("close: ") + std::strerror(errno);
            return false;
        }
        fd = -1;
        if (::rename(tmp.c_str(), path.c_str()) != 0) {
            err = std::string("rename: ") + std::strerror(errno);
            return false;
        }
        bytes = total;
        tmp.clear();
        return true;
    }
    void abort() {
        if (fd >= 0) ::close(fd);
        fd = -1;
        if (!tmp.empty()) ::unlink(tmp.c_str());
        tmp.clear();
    }
};

struct SessionIn {
    int fd = -1;
    uint64_t hash = kSessionFnvSeed;
    uint64_t total = 0;

    ~SessionIn() { if (fd >= 0) ::close(fd); }

    bool open(const std::string& path, std::string& err) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) {
            err = std::string("open: ") + std::strerror(errno);
            return false;
        }
        struct stat st;
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
            err = "not a regular file";
            return false;
        }
        total = (uint64_t) st.st_size;
        return true;
    }
    bool get_raw(void* p, size_t n, std::string& err) {
        uint8_t* q = static_cast<uint8_t*>(p);
        while (n > 0) {
            const ssize_t r = ::read(fd, q, n);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) {
                err = "truncated session file";
                return false;
            }
            q += r;
            n -= (size_t) r;
        }
        return true;
    }
    bool get(void* p, size_t n, std::string& err) {
        if (!get_raw(p, n, err)) return false;
        hash = session_fnv(hash, p, n);
        return true;
    }
    bool finish(std::string& err) {
        uint64_t h = 0;
        char end[8] = {};
        if (!get_raw(&h, sizeof h, err) || !get_raw(end, sizeof end, err)) return false;
        static const char want[8] = {'P', 'O', 'L', 'Y', 'E', 'N', 'D', '!'};
        if (std::memcmp(end, want, sizeof want) != 0) {
            err = "not a poly session file (bad trailer)";
            return false;
        }
        if (h != hash) {
            err = "session file checksum mismatch";
            return false;
        }
        return true;
    }
};
#endif

inline bool session_disk_room(const std::string& path, uint64_t need, std::string& err) {
#ifndef _WIN32
    const size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : (slash == 0 ? "/" : path.substr(0, slash));
    struct statvfs vfs;
    if (::statvfs(dir.c_str(), &vfs) != 0) {
        err = std::string("statvfs: ") + std::strerror(errno);
        return false;
    }
    const uint64_t freeb = (uint64_t) vfs.f_bavail * (uint64_t) vfs.f_frsize;
    if (freeb < need) {
        err = "not enough free disk space for the session file";
        return false;
    }
#else
    (void) path; (void) need; (void) err;
#endif
    return true;
}

}  // namespace detail

template<class E>
class Conversations {
public:
    Conversations(E& engine, CacheOptions o)
        : e_(engine), o_(std::move(o)), slots_((size_t) std::max(o_.park_slots, 0) + 1) {}

    /// The tokens the engine has read.
    const std::vector<int32_t>& seq() const { return slots_[cur_].seq; }

    /// Leaves the engine at the furthest point of `ids` that can be come back to, and returns that position.
    int64_t resume(const std::vector<int32_t>& ids) {
        const int64_t n = (int64_t) ids.size();
        turn_ = -1;
        size_t best = cur_;
        int64_t at = 0, same = 0;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const int64_t c = common(slots_[i].seq, ids), r = reach(i, c, n);
            if (r > at || (r == at && i == cur_)) best = i, at = r, same = c;
        }
        ++clock_;
        if (at == 0) {
            // nothing kept starts like this
            if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
            slots_[cur_] = Slot{};
            e_.reset();
        } else if (best != cur_ || at < len(cur())) {
            if (best != cur_) enter(best, at);
            // The conversation is left at `at`.  If the prompt the client sent for it went on from there, the
            // client may send that again, so the conversation is parked and this one goes on in a copy.
            if (same >= cur().prompt || !branch(at)) cut(cur(), at);
            e_.load_state(state_at(cur(), at), at);
        }
        cur().stamp = clock_;
        return at;
    }

    /// One more checkpoint of the prompt resume() took up: where its chat's last turn begins, which the server
    /// knows from the template.  The chat's next request renders the conversation up to there as this one does,
    /// but this turn's header and reply not always in the same tokens (a template drops the reply's thinking), so
    /// the checkpoint before the prompt's last token is seldom where it goes on from.  A place the read does not
    /// pass adds none.
    void turn_at(int64_t at) { turn_ = at; }

    /// Of `want` tokens from position `at`, how many to read before the next checkpoint is due.
    int64_t until_checkpoint(int64_t at, int64_t want) const {
        if (o_.checkpoints < 1) return want;
        if (o_.every > 0) want = std::min(want, (at / o_.every + 1) * o_.every - at);
        return turn_ > at ? std::min(want, turn_ - at) : want;
    }

    /// Before the engine reads the token at `at`; `last` says it is the prompt's last.  Takes a checkpoint if one is
    /// due.  One sits before the last token of every prompt, so coming back to it leaves a token to read and the
    /// logits that come with it.
    void before(int64_t at, bool last) {
        if (o_.checkpoints < 1 || at < 1 || (!last && at != turn_ && (o_.every < 1 || at % o_.every != 0))) return;
        if (const BlobPtr b = mark(at)) e_.save_state(b->p.get());
    }

    /// For an engine that reads past checkpoints in one go, of n tokens from `at`: the checkpoints due after the
    /// read's first token.  `out` gets, for each, how many of the read's tokens come before it and where its
    /// state_bytes() go; null where no state is wanted, and the engine still breaks its work there as if one
    /// were.  The engine writes a state when it has it.  The checkpoints are the conversation's once read() has
    /// the read's tokens, and not at all when the read is given up (rewind()).
    ///
    /// A state is wanted for the checkpoints the conversation would still have, had each been taken in turn as
    /// before() takes one.  A long prompt has more due than are kept, and a state can be hundreds of MiB.
    void inside(int64_t at, int64_t n, std::vector<std::pair<int64_t, uint8_t*>>& out) {
        out.clear();
        due_.clear();
        after_.clear();
        if (o_.checkpoints < 1) return;
        std::vector<int64_t> due;
        if (o_.every > 0)
            for (int64_t p = (at / o_.every + 1) * o_.every; p < at + n; p += o_.every) due.push_back(p);
        if (turn_ > at && turn_ < at + n && (o_.every < 1 || turn_ % o_.every != 0))
            due.insert(std::upper_bound(due.begin(), due.end(), turn_), turn_);
        if (due.empty()) return;
        for (const Checkpoint& m : cur().marks) after_.push_back(m.pos);
        for (const int64_t p : due) {
            after_.push_back(p);
            while (after_.size() > (size_t) o_.checkpoints) after_.erase(after_.begin() + (std::ptrdiff_t) drop_of(after_));
        }
        for (const int64_t p : due) {
            const bool stays = std::find(after_.begin(), after_.end(), p) != after_.end();
            due_.push_back({p, stays ? blob(e_.state_bytes()) : nullptr});
            out.emplace_back(p - at, due_.back().state ? due_.back().state->p.get() : nullptr);
        }
    }

    /// The engine has read these tokens.  A read that inside() was asked about is done with them, and its
    /// checkpoints take their places.
    void read(const int32_t* t, int64_t n) {
        Slot& s = cur();
        s.seq.insert(s.seq.end(), t, t + n);
        if (due_.empty()) return;
        const auto stays = [&](const Checkpoint& m) {
            return m.state && std::find(after_.begin(), after_.end(), m.pos) != after_.end();
        };
        std::vector<Checkpoint> marks;
        for (const Checkpoint& m : s.marks)
            if (stays(m)) marks.push_back(m);
        for (const Checkpoint& m : due_)
            if (stays(m)) marks.push_back(m);
        s.marks = std::move(marks);
        due_.clear();
    }

    /// A read was given up part way: what it left in the engine is not the conversation's, and the checkpoints
    /// inside it are not taken.  The conversation goes back to its last checkpoint, which the engine is left
    /// at, or is forgotten when it has none.  Returns where the conversation is then.
    int64_t rewind() {
        due_.clear();
        Slot& s = cur();
        if (s.marks.empty()) {
            slots_[cur_] = Slot{};
            e_.reset();
            return 0;
        }
        const int64_t at = s.marks.back().pos;
        cut(s, at);
        e_.load_state(s.marks.back().state->p.get(), at);
        return at;
    }
    /// The first n tokens of the conversation are the prompt the client sent.
    void prompt_was(int64_t n) { cur().prompt = std::min(n, len(cur())); }
    /// After an engine error: the conversation being read is lost.  The caller resets the engine.
    void drop() {
        slots_[cur_] = Slot{};
        due_.clear();
    }

    /// The engine's sequence is about to hold another conversation, read elsewhere (a batch slot's copy): the one
    /// it holds is parked, as resume() parks it, or forgotten.  adopt() then says what it holds.
    void leave() {
        if (!cur().seq.empty() && park(slots_.size())) cur_ = take(cur_);
        slots_[cur_] = Slot{};
        due_.clear();
    }
    /// The engine holds `seq` now, with the state at its end: a prompt that goes on from there reads none of it.
    void adopt(std::vector<int32_t> seq) {
        Slot& s = cur();
        s = Slot{};
        s.seq = std::move(seq);
        s.prompt = len(s);
        s.stamp = ++clock_;
    }

    /// The parked conversations, the bytes they hold, and how many were pushed out so far.
    size_t parked() const { return held(slots_.size()).count; }
    size_t parked_bytes() const { return held(slots_.size()).bytes; }
    size_t evictions() const { return evictions_; }

    /// Writes the current conversation, up to its last checkpoint, to `path`.  `identity` binds the file to the
    /// model; a restore with another identity is refused.  On success `tokens` and `bytes` describe the file.
    bool save_session(const std::string& path, const std::string& identity, uint64_t& tokens, uint64_t& bytes,
                      std::string& err) {
#ifdef _WIN32
        (void) path; (void) identity; (void) tokens; (void) bytes;
        err = "session files are not supported on this Windows build";
        return false;
#else
        if (o_.checkpoints < 1) {
            err = "session files need --prompt-cache > 0";
            return false;
        }
        Slot& s = cur();
        if (s.marks.empty() || s.marks.back().pos <= 0) {
            err = "no complete session to save";
            return false;
        }
        const int64_t end = s.marks.back().pos;
        const size_t rb = e_.row_bytes(), sb = e_.state_bytes();
        for (const Checkpoint& m : s.marks) {
            if (!m.state || m.state->bytes != sb) {
                err = "checkpoint state has the wrong size";
                return false;
            }
        }
        BlobPtr all = blob((size_t) end * rb);
        if (!all) {
            err = "out of memory";
            return false;
        }
        for (const Rows& r : s.rows) {
            if (r.first >= end) continue;
            const int64_t n = std::min(r.keep, end - r.first);
            if (n > 0) {
                std::memcpy(all->p.get() + (size_t) r.first * rb, r.blob->p.get(), (size_t) n * rb);
            }
        }
        const int64_t from = saved(s);
        if (from < end) {
            try {
                e_.save_rows(from, end - from, all->p.get() + (size_t) from * rb);
            } catch (const std::exception& ex) {
                err = std::string("capture: ") + ex.what();
                return false;
            }
        }
        // Serialize a copy ending at the last checkpoint; the live conversation and the engine stay where they are.
        Slot save = s;
        save.seq.resize((size_t) end);
        save.prompt = std::min<int64_t>(save.prompt, end);
        save.rows.clear();
        save.rows.push_back({all, 0, end, end});
        // One state, the deepest checkpoint: the others would copy the same fixed recurrent state again.  A restore
        // resumes from here and re-reads at most the tokens after this checkpoint, as --strata's files do.
        if (save.marks.size() > 1) save.marks = {save.marks.back()};

        uint64_t total = 84 + (uint64_t) end * sizeof(int32_t) + 16 + 8 + (uint64_t) end * rb;
        for (const Checkpoint& m : save.marks) total += 16 + m.state->bytes;
        if (!detail::session_disk_room(path, total + ((uint64_t) o_.session_min_free_mib << 20), err)) {
            return false;
        }
        detail::SessionOut out;
        if (!out.open(path, err)) return false;
        const char magic[16] = "USTRPOLYSESSION";
        const uint32_t version = 1;
        uint64_t id = detail::session_fnv(detail::kSessionFnvSeed, identity.data(), identity.size());
        const uint64_t nctx = (uint64_t) e_.max_context();
        id = detail::session_fnv(id, &nctx, sizeof nctx);
        id = detail::session_fnv(id, &rb, sizeof rb);
        id = detail::session_fnv(id, &sb, sizeof sb);
        const uint64_t seq_len = (uint64_t) end;
        const uint64_t prompt = (uint64_t) save.prompt;
        const uint64_t nmarks = save.marks.size();
        const uint64_t rows_bytes = (uint64_t) end * rb;
        bool ok = out.put(magic, sizeof magic, err) && out.put(&version, sizeof version, err) &&
                  out.put(&id, sizeof id, err) && out.put(&nctx, sizeof nctx, err) &&
                  out.put(&rb, sizeof rb, err) && out.put(&sb, sizeof sb, err) &&
                  out.put(&seq_len, sizeof seq_len, err) && out.put(&prompt, sizeof prompt, err) &&
                  out.put(&nmarks, sizeof nmarks, err) && out.put(&rows_bytes, sizeof rows_bytes, err);
        if (ok) ok = out.put(save.seq.data(), save.seq.size() * sizeof(int32_t), err);
        for (const Checkpoint& m : save.marks) {
            if (!ok) break;
            const uint64_t pos = (uint64_t) m.pos;
            const uint64_t n = (uint64_t) m.state->bytes;
            ok = out.put(&pos, sizeof pos, err) && out.put(&n, sizeof n, err) &&
                 out.put(m.state->p.get(), m.state->bytes, err);
        }
        if (ok) ok = out.put(save.rows[0].blob->p.get(), save.rows[0].blob->bytes, err);
        if (!ok) return false;
        if (!out.finish(path, err, bytes)) return false;
        tokens = seq_len;
        return true;
#endif
    }

    /// Loads a conversation written by save_session() into the engine.  The file is fully validated before the
    /// engine is touched; a bad file leaves the engine as it was.
    bool restore_session(const std::string& path, const std::string& identity, uint64_t& tokens, uint64_t& bytes,
                         std::string& err) {
#ifdef _WIN32
        (void) path; (void) identity; (void) tokens; (void) bytes;
        err = "session files are not supported on this Windows build";
        return false;
#else
        if (o_.checkpoints < 1) {
            err = "session files need --prompt-cache > 0";
            return false;
        }
        detail::SessionIn in;
        if (!in.open(path, err)) return false;
        char magic[16] = {};
        uint32_t version = 0;
        uint64_t id = 0, nctx = 0, rb = 0, sb = 0, seq_len = 0, prompt = 0, nmarks = 0, rows_bytes = 0;
        const char want_magic[16] = "USTRPOLYSESSION";
        bool ok = in.get(magic, sizeof magic, err) && in.get(&version, sizeof version, err) &&
                  in.get(&id, sizeof id, err) && in.get(&nctx, sizeof nctx, err) &&
                  in.get(&rb, sizeof rb, err) && in.get(&sb, sizeof sb, err) &&
                  in.get(&seq_len, sizeof seq_len, err) && in.get(&prompt, sizeof prompt, err) &&
                  in.get(&nmarks, sizeof nmarks, err) && in.get(&rows_bytes, sizeof rows_bytes, err);
        if (!ok) return false;
        if (std::memcmp(magic, want_magic, sizeof magic) != 0 || version != 1) {
            err = "not a poly session file (bad header)";
            return false;
        }
        if (nctx != (uint64_t) e_.max_context() || rb != e_.row_bytes() || sb != e_.state_bytes()) {
            err = "session file was written by another model or engine configuration";
            return false;
        }
        if (seq_len == 0 || seq_len > nctx || nmarks == 0 || nmarks > 4096 || rows_bytes != seq_len * rb) {
            err = "session file has invalid sizes";
            return false;
        }
        uint64_t want_id = detail::session_fnv(detail::kSessionFnvSeed, identity.data(), identity.size());
        want_id = detail::session_fnv(want_id, &nctx, sizeof nctx);
        want_id = detail::session_fnv(want_id, &rb, sizeof rb);
        want_id = detail::session_fnv(want_id, &sb, sizeof sb);
        if (id != want_id) {
            err = "session file belongs to another model";
            return false;
        }
        Slot s;
        s.seq.resize((size_t) seq_len);
        if (!in.get(s.seq.data(), s.seq.size() * sizeof(int32_t), err)) return false;
        int64_t last = 0;
        for (uint64_t i = 0; i < nmarks; ++i) {
            uint64_t pos = 0, n = 0;
            if (!in.get(&pos, sizeof pos, err) || !in.get(&n, sizeof n, err)) return false;
            if (n != sb || pos <= 0 || pos > seq_len || (int64_t) pos <= last) {
                err = "session file has an invalid checkpoint";
                return false;
            }
            BlobPtr b = blob((size_t) n);
            if (!b) {
                err = "out of memory";
                return false;
            }
            if (!in.get(b->p.get(), b->bytes, err)) return false;
            s.marks.push_back({(int64_t) pos, std::move(b)});
            last = (int64_t) pos;
        }
        BlobPtr rows = blob((size_t) rows_bytes);
        if (!rows) {
            err = "out of memory";
            return false;
        }
        if (!in.get(rows->p.get(), rows->bytes, err) || !in.finish(err)) return false;
        if (s.marks.back().pos != (int64_t) seq_len) {
            err = "session file does not end at its last checkpoint";
            return false;
        }
        s.prompt = std::min<int64_t>((int64_t) prompt, (int64_t) seq_len);
        s.rows.push_back({rows, 0, (int64_t) seq_len, (int64_t) seq_len});
        bytes = in.total;
        try {
            e_.reset();
        } catch (const std::exception& ex) {
            err = std::string("reset: ") + ex.what();
            return false;
        }
        slots_[cur_] = std::move(s);
        try {
            Rows& r = slots_[cur_].rows[0];
            e_.load_rows(r.blob->p.get(), 0, (int64_t) seq_len, 0, (int64_t) seq_len);
            e_.load_state(slots_[cur_].marks.back().state->p.get(), (int64_t) seq_len);
        } catch (const std::exception& ex) {
            slots_[cur_] = Slot{};
            try { e_.reset(); } catch (const std::exception&) {}
            err = std::string("load: ") + ex.what();
            return false;
        }
        due_.clear();
        turn_ = -1;
        tokens = seq_len;
        return true;
#endif
    }

private:
    using Clock = std::chrono::steady_clock;

    struct Blob {
        std::unique_ptr<uint8_t[]> p;
        size_t bytes = 0;
    };
    using BlobPtr = std::shared_ptr<const Blob>;
    struct Checkpoint {
        int64_t pos;
        BlobPtr state;
    };
    // The blob holds `count` rows from row `first`; the first `keep` of them are this conversation's.
    struct Rows {
        BlobPtr blob;
        int64_t first, count, keep;
    };
    struct Slot {
        std::vector<int32_t> seq;
        int64_t prompt = 0;             // seq's first tokens that the client sent
        std::vector<Checkpoint> marks;  // by position
        std::vector<Rows> rows;         // back to back from row 0; a parked conversation has all of its rows
        uint64_t stamp = 0;
    };
    struct Held {
        size_t count = 0, bytes = 0;
    };

    Slot& cur() { return slots_[cur_]; }
    const Slot& cur() const { return slots_[cur_]; }
    static int64_t len(const Slot& s) { return (int64_t) s.seq.size(); }
    static int64_t saved(const Slot& s) { return s.rows.empty() ? 0 : s.rows.back().first + s.rows.back().keep; }

    static int64_t common(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return (int64_t) n;
    }

    // How far into a prompt of n tokens slot i gets it, given that they start with `same` tokens alike.
    int64_t reach(size_t i, int64_t same, int64_t n) const {
        const Slot& s = slots_[i];
        if (i == cur_ && same > 0 && same == len(s) && n > same) return same;   // the engine is right there
        int64_t at = 0;
        for (const Checkpoint& m : s.marks)
            if (m.pos <= std::min(same, n - 1)) at = m.pos;
        return at;
    }

    static const uint8_t* state_at(const Slot& s, int64_t at) {
        for (const Checkpoint& m : s.marks)
            if (m.pos == at) return m.state->p.get();
        return nullptr;   // resume() only goes to positions reach() found
    }

    static BlobPtr blob(size_t bytes) {
        auto b = std::make_shared<Blob>();
        b->p.reset(new (std::nothrow) uint8_t[bytes]);
        b->bytes = bytes;
        return b->p ? b : nullptr;
    }

    // A checkpoint of the current conversation at `at`, its state still to be written.  Null: there is one there,
    // or no memory for it.
    BlobPtr mark(int64_t at) {
        Slot& s = cur();
        if (!s.marks.empty() && s.marks.back().pos == at) return nullptr;
        const BlobPtr b = blob(e_.state_bytes());
        if (!b) return nullptr;
        s.marks.push_back({at, b});
        while (s.marks.size() > (size_t) o_.checkpoints) {
            std::vector<int64_t> pos;
            for (const Checkpoint& m : s.marks) pos.push_back(m.pos);
            s.marks.erase(s.marks.begin() + (std::ptrdiff_t) drop_of(pos));
        }
        return b;
    }

    // Of more checkpoints than are kept, the one that goes: the one whose neighbours are closest together.  The
    // last two stay, the end of this prompt and of the one before being where a client most often comes back to.
    static size_t drop_of(const std::vector<int64_t>& pos) {
        size_t drop = 0;
        int64_t gap = INT64_MAX;
        for (size_t k = 0; k + 2 < pos.size(); ++k) {
            const int64_t g = pos[k + 1] - (k ? pos[k - 1] : 0);
            if (g < gap) gap = g, drop = k;
        }
        return drop;
    }

    static void cut(Slot& s, int64_t at) {
        if (at >= len(s)) return;
        s.seq.resize((size_t) at);
        s.prompt = std::min(s.prompt, at);
        while (!s.marks.empty() && s.marks.back().pos > at) s.marks.pop_back();
        while (!s.rows.empty() && s.rows.back().first >= at) s.rows.pop_back();
        if (!s.rows.empty()) s.rows.back().keep = std::min(s.rows.back().keep, at - s.rows.back().first);
    }

    // The parked conversations and what they hold; a blob several of them share counts once.  The current one
    // counts when `with` is its slot.
    Held held(size_t with) const {
        Held h;
        std::vector<const Blob*> seen;
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot& s = slots_[i];
            if (s.seq.empty() || (i == cur_ && i != with)) continue;
            ++h.count;
            for (const Checkpoint& m : s.marks) seen.push_back(m.state.get());
            for (const Rows& r : s.rows) seen.push_back(r.blob.get());
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (const Blob* b : seen) h.bytes += b->bytes;
        return h;
    }

    // Makes the parked conversations, the current one among them and `extra` bytes more, fit the budget and the
    // slots.  The least recently used goes first, whatever its length: a long conversation can lose its place to
    // several short ones.  Slot `keep` stays; it is the one about to take the engine, if any, so it frees a slot.
    // False: it does not fit, or the host cannot spare the bytes.
    bool fit(size_t extra, size_t keep) {
        const size_t budget = (size_t) o_.park_mib << 20;
        for (;;) {
            const Held h = held(cur_);
            const size_t leaving = keep < slots_.size() && keep != cur_ && !slots_[keep].seq.empty() ? 1 : 0;
            if (h.bytes + extra <= budget && h.count - leaving <= (size_t) o_.park_slots) break;
            size_t out = slots_.size();
            for (size_t i = 0; i < slots_.size(); ++i)
                if (i != cur_ && i != keep && !slots_[i].seq.empty() &&
                    (out == slots_.size() || slots_[i].stamp < slots_[out].stamp))
                    out = i;
            if (out == slots_.size()) return false;
            slots_[out] = Slot{};
            ++evictions_;
        }
        return extra == 0 || !o_.admit || o_.admit(extra);
    }

    // Copies the rows the current conversation still has only in the engine, up to its last checkpoint; what it
    // read after that cannot be come back to and is dropped.  False: it cannot be kept.
    bool park(size_t keep) {
        Slot& s = cur();
        if (o_.park_mib < 1 || o_.park_slots < 1 || s.marks.empty() || s.marks.back().pos < o_.park_min) return false;
        const Clock::time_point t0 = Clock::now();
        const int64_t end = s.marks.back().pos, from = saved(s);
        cut(s, end);
        const size_t bytes = (size_t) (end - from) * e_.row_bytes();
        const BlobPtr b = from < end && fit(bytes, keep) ? blob(bytes) : nullptr;
        if (from < end && !b) {
            say("skipped", end, t0, slots_.size());
            return false;
        }
        if (b) {
            e_.save_rows(from, end - from, b->p.get());
            s.rows.push_back({b, from, end - from, end - from});
        }
        if (!fit(0, keep)) return false;
        say("parked", end, t0, cur_);
        return true;
    }

    // An empty slot other than `keep`.
    size_t take(size_t keep) {
        size_t out = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i == keep) continue;
            if (slots_[i].seq.empty()) return i;
            if (out == slots_.size() || slots_[i].stamp < slots_[out].stamp) out = i;
        }
        slots_[out] = Slot{};   // park() left one empty, so this is not reached
        ++evictions_;
        return out;
    }

    // Slot i takes the engine, with its rows before `need` back in the caches.  The conversation that leaves is
    // parked if it can be.  Rows that both have from the same blob are in the caches already.
    void enter(size_t i, int64_t need) {
        const Clock::time_point t0 = Clock::now();
        const Slot& in = slots_[i];
        int64_t have = 0;
        for (size_t k = 0; k < in.rows.size() && k < cur().rows.size() && in.rows[k].blob == cur().rows[k].blob; ++k) {
            have = in.rows[k].first + std::min(in.rows[k].keep, cur().rows[k].keep);
            if (in.rows[k].keep != cur().rows[k].keep) break;
        }
        if (cur().seq.empty() || !park(i)) slots_[cur_] = Slot{};
        for (const Rows& r : in.rows) {
            const int64_t a = std::max(r.first, have), b = std::min(r.first + r.keep, need);
            if (a < b) e_.load_rows(r.blob->p.get(), r.first, r.count, a, b - a);
        }
        cur_ = i;
        say("restored", need, t0, slots_.size());
    }

    // Parks the current conversation whole and goes on at `at` in a copy of its start, which shares what is
    // saved.  False: it cannot be parked.
    bool branch(int64_t at) {
        const size_t from = cur_;
        if (!park(slots_.size())) return false;
        const size_t to = take(from);
        const Slot& a = slots_[from];
        Slot b;
        b.seq.assign(a.seq.begin(), a.seq.begin() + (std::ptrdiff_t) at);
        b.prompt = std::min(a.prompt, at);
        for (const Checkpoint& m : a.marks)
            if (m.pos <= at) b.marks.push_back(m);
        for (const Rows& r : a.rows)
            if (r.first < at) b.rows.push_back({r.blob, r.first, r.count, std::min(r.keep, at - r.first)});
        slots_[to] = std::move(b);
        cur_ = to;
        return true;
    }

    // `with` is the current slot when its conversation is the one just parked.
    void say(const char* what, int64_t tokens, Clock::time_point t0, size_t with) const {
        if (!o_.log) return;
        const Held h = held(with);
        std::fprintf(o_.log,
                     "strata-poly: conversation cache: %s %lld tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu\n",
                     what, (long long) tokens, std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     h.count, h.bytes, evictions_);
    }

    E& e_;
    CacheOptions o_;
    std::vector<Slot> slots_;   // the current conversation and the parked ones
    size_t cur_ = 0;
    uint64_t clock_ = 0;
    size_t evictions_ = 0;
    std::vector<Checkpoint> due_;   // inside()'s checkpoints, until read() makes them the conversation's
    std::vector<int64_t> after_;    // where the conversation's checkpoints are once that read is done
    int64_t turn_ = -1;             // turn_at()'s place in the prompt being read; -1: none
};

}  // namespace strata::poly
