package main

/*
File:  ll-hls-origin-example.go

Copyright 2019-2020 Apple Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
)

type logValues struct {
	StartTime     time.Time
	Client        string
	Method        string
	Protocol      string
	RequestURI    string
	Scheme        string
	HostHdr       string
	BlockDuration time.Duration // Duration spent blocked waiting for the resource to become available
	TotalDuration time.Duration // Total duration from the start of receiving the request until the data was off to the NIC
	Size          uint64
	StatusCode    int
}

const (
	index               = "prog_index.m3u8"
	playListEndPoint    = "lowLatencyHLS.m3u8"
	segmentEndPoint     = "lowLatencySeg"
	serverVersionString = "ll-hls/golang/0.1"
	canSkipUntil        = 6
	seqParamQName       = "_HLS_msn"
	partParamQName      = "_HLS_part"
	skipParamQName      = "_HLS_skip"
)

var (
	httpAddr = flag.String("http", ":8443", "Listen address")
	dir      = flag.String("dir", "", "Root dir with hls files")
	certDir  = flag.String("certdir", "", "Dir with server.crt, and server.key files")
)

// SimpleMediaPlaylist is a simple struct that represents everything needed to handle a Low Latency Media Playlist
type SimpleMediaPlaylist struct {
	TargetDuration      time.Duration     // #EXT-X-TARGETDURATION:4
	Version             uint64            // #EXT-X-VERSION:3
	PartTargetDuration  time.Duration     // #EXT-X-PART-INF:PART-TARGET=1.004000
	MediaSequenceNumber uint64            // #EXT-X-MEDIA-SEQUENCE:339
	Segments            []FullSegment     // The segment list of the mediaplaylist
	NextMSNIndex        uint64            // The index to be used for the next full segment
	NextPartIndex       uint64            // The index to be used for the next partial segment
	MaxPartIndex        uint64            // To determine when to "roll over" on the NextPartIndex
	PreloadHints        map[string]string // A map[<TYPE>]URI
}

// SimpleSegment is a struct that represents a HLS Segment
type SimpleSegment struct {
	Duration    float64  // #EXTINF:3.96667,
	URI         string   // fileSequence5.ts
	ExtraLines  []string // #EXT-X-PROGRAM-DATE-TIME:2019-11-08T22:41:10.072Z and many more
	Independent bool     // INDEPENDENT=YES
}

// FullSegment is a segment with a set of children
type FullSegment struct {
	Self  SimpleSegment   // This contains the information for the full segment (if complete)
	Parts []SimpleSegment // An array of part segments that this full is made up off
}

// LastMSN returns the last MSN index
func (mp *SimpleMediaPlaylist) LastMSN() uint64 {
	if mp.NextPartIndex == 0 {
		return mp.NextMSNIndex - 1
	}
	return mp.NextMSNIndex
}

// LastPart returns the last PART index
func (mp *SimpleMediaPlaylist) LastPart() uint64 {
	if mp.NextPartIndex == 0 {
		return mp.MaxPartIndex - 1
	}
	return mp.NextPartIndex - 1
}

func newFullSegment() FullSegment {
	return FullSegment{
		Self: SimpleSegment{
			URI: "",
		},
		Parts: make([]SimpleSegment, 0),
	}
}

// EncodeWithSkip encodes the struct to the string playlist update
func (mp *SimpleMediaPlaylist) EncodeWithSkip(skipUntil uint64) string {
	return mp.encode(skipUntil)
}

// Encode the struct to the string full playlist
func (mp *SimpleMediaPlaylist) Encode() string {
	return mp.encode(0)
}

func (mp *SimpleMediaPlaylist) encode(skipUntil uint64) string {
	totalDurationOfPlaylist := mp.TargetDuration.Seconds() * float64(len(mp.Segments))
	skipDuration := 0.0
	skippedSegments := uint64(0)
	version := mp.Version
	if skipUntil > 0 {
		skipDuration = totalDurationOfPlaylist - float64(skipUntil+2)*mp.TargetDuration.Seconds()
		skippedSegments = uint64(math.Floor(skipDuration / mp.TargetDuration.Seconds()))
		version = 9
	}
	out := "#EXTM3U\n"
	out += fmt.Sprintf("#EXT-X-TARGETDURATION:%s\n", strconv.FormatFloat(mp.TargetDuration.Seconds(), 'f', -1, 64))
	out += fmt.Sprintf("#EXT-X-VERSION:%d\n", version)
	out += fmt.Sprintf("#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,CAN-SKIP-UNTIL=%1.0f,PART-HOLD-BACK=%1.3f\n", float64(canSkipUntil)*mp.TargetDuration.Seconds(), 3*mp.PartTargetDuration.Seconds())
	out += fmt.Sprintf("#EXT-X-PART-INF:PART-TARGET=%s\n", strconv.FormatFloat(mp.PartTargetDuration.Seconds(), 'f', 6, 64))
	out += fmt.Sprintf("#EXT-X-MEDIA-SEQUENCE:%d\n", mp.MediaSequenceNumber)
	if skippedSegments > 0 {
		out += fmt.Sprintf("#EXT-X-SKIP:SKIPPED-SEGMENTS=%d\n", skippedSegments)
	}

	durationSkipped := 0.0
	for _, fullSeg := range mp.Segments {
		if durationSkipped < skipDuration {
			durationSkipped += mp.TargetDuration.Seconds()
			continue
		}
		for _, eLine := range fullSeg.Self.ExtraLines {
			if eLine == "" {
				continue
			}
			out += fmt.Sprintf("%s\n", eLine)
		}
		if len(fullSeg.Parts) > 0 {
			for _, partSeg := range fullSeg.Parts {
				fileExt := filepath.Ext(partSeg.URI)
				if partSeg.Independent {
					out += fmt.Sprintf("#EXT-X-PART:DURATION=%s,INDEPENDENT=YES,URI=\"%s%s?segment=%s\"\n", strconv.FormatFloat(partSeg.Duration, 'f', 5, 64), segmentEndPoint, fileExt, partSeg.URI)
				} else {
					out += fmt.Sprintf("#EXT-X-PART:DURATION=%s,URI=\"%s%s?segment=%s\"\n", strconv.FormatFloat(partSeg.Duration, 'f', 5, 64), segmentEndPoint, fileExt, partSeg.URI)
				}
			}
		}
		if fullSeg.Self.URI != "" {
			out += fmt.Sprintf("#EXTINF:%s,\n", strconv.FormatFloat(fullSeg.Self.Duration, 'f', 5, 32))
			out += fmt.Sprintf("%s\n", fullSeg.Self.URI)
		}
	}
	if mp.PreloadHints != nil {
		for hintType, hintURI := range mp.PreloadHints {
			fileExt := filepath.Ext(hintURI)
			out += fmt.Sprintf("#EXT-X-PRELOAD-HINT:TYPE=%s,URI=\"%s%s?segment=%s\"\n", hintType, segmentEndPoint, fileExt, hintURI)
		}
	}
	return out
}

// Decode a simple m3u8 media playlist as generated by mediastreamsegmenter in the Beta LL HLS Tools package.
// All lines that are not needed for this example are stored in segment.ExtraLines and just preserved on Decode/Encode
func Decode(reader io.Reader) (*SimpleMediaPlaylist, error) {
	mp := SimpleMediaPlaylist{
		Segments: make([]FullSegment, 0),
	}
	var err error
	currentFullSegment := newFullSegment()

	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case line == "#EXTM3U":
		case strings.HasPrefix(line, "#EXT-X-TARGETDURATION:"):
			stringVal := strings.Split(line, ":")[1]
			mp.TargetDuration, err = time.ParseDuration(stringVal + "s")
		case strings.HasPrefix(line, "#EXT-X-VERSION:"):
			stringVal := strings.Split(line, ":")[1]
			mp.Version, err = strconv.ParseUint(stringVal, 0, 64)
		case strings.HasPrefix(line, "#EXT-X-PART-INF:"):
			stringVal := strings.Split(line, "=")[1]
			mp.PartTargetDuration, err = time.ParseDuration(stringVal + "s")
		case strings.HasPrefix(line, "#EXT-X-MEDIA-SEQUENCE:"):
			stringVal := strings.Split(line, ":")[1]
			mp.MediaSequenceNumber, err = strconv.ParseUint(stringVal, 0, 64)
			mp.NextMSNIndex = mp.MediaSequenceNumber
		case strings.HasPrefix(line, "#EXTINF:"):
			stringVal := strings.Split(line, ":")[1]
			stringVal = strings.Split(stringVal, ",")[0]
			currentFullSegment.Self.Duration, err = strconv.ParseFloat(stringVal, 64)
		case line != "" && !strings.HasPrefix(line, "#"):
			// The URI line is the last line for the segment, add it to the playlist and create a new empty one
			currentFullSegment.Self.URI = line
			mp.Segments = append(mp.Segments, currentFullSegment)
			mp.NextMSNIndex++
			mp.NextPartIndex = 0
			currentFullSegment = newFullSegment()
		case strings.HasPrefix(line, "#EXT-X-PART:"):
			// Parts get added to a the full.Parts array
			var part SimpleSegment
			params := strings.Split(line[12:], ",")
			for _, param := range params {
				parts := strings.SplitN(param, "=", 2)
				key := parts[0]
				value := parts[1]
				switch key {
				case "DURATION":
					part.Duration, err = strconv.ParseFloat(value, 64)
				case "URI":
					part.URI = strings.ReplaceAll(value, "\"", "")
				case "INDEPENDENT":
					if value == "YES" {
						part.Independent = true
					}
				}
			}
			currentFullSegment.Parts = append(currentFullSegment.Parts, part)
			mp.NextPartIndex++
			if mp.MaxPartIndex < mp.NextPartIndex {
				mp.MaxPartIndex = mp.NextPartIndex
			}
		case strings.HasPrefix(line, "#EXT-X-PRELOAD-HINT:"):
			mp.PreloadHints = make(map[string]string)
			params := strings.Split(line[20:], ",")
			hintType := ""
			hintURI := ""
			for _, param := range params {
				parts := strings.SplitN(param, "=", 2)
				key := parts[0]
				value := parts[1]
				switch key {
				case "TYPE":
					hintType = value
				case "URI":
					hintURI = strings.ReplaceAll(value, "\"", "")
				}
			}
			mp.PreloadHints[hintType] = hintURI
		default: // Only fulls get ExtraLines - don't touch them, just add them
			currentFullSegment.Self.ExtraLines = append(currentFullSegment.Self.ExtraLines, line)

		}

		if err != nil {
			log.Printf("Error parsing -%s- :%s ", line, err)
			return nil, err
		}
	}
	if currentFullSegment.Self.URI == "" {
		// Need to add the last parts, so add the last current. It has URI="" so it's Self will be ignored
		mp.Segments = append(mp.Segments, currentFullSegment)
	}
	return &mp, nil
}

func getMediaPlaylist(file string) (*SimpleMediaPlaylist, error) {
	fh, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	mediaPlaylist, err := Decode(fh)
	if err != nil {
		return nil, err
	}
	return mediaPlaylist, nil
}

func waitForPlaylistWithSequenceNumber(file string, seqNo uint64, partNo uint64) (time.Duration, *SimpleMediaPlaylist, error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return 0, nil, err
	}
	defer watcher.Close()

	err = watcher.Add(file)
	if err != nil {
		return 0, nil, err
	}
	defer watcher.Remove(file)

	mediaPlaylist, err := getMediaPlaylist(file)
	if err != nil {
		return 0, nil, err
	}
	if seqNo == 0 || // There were no _HLS_ parameters
		mediaPlaylist.LastMSN() > seqNo ||
		(mediaPlaylist.LastMSN() == seqNo && mediaPlaylist.LastPart() >= partNo) {
		return 0, mediaPlaylist, nil
	}
	// If a client supplies an _HLS_msn parameter greater than the Media Sequence Number of the last segment in the Playlist plus 2 .. return 400
	if seqNo > mediaPlaylist.LastMSN()+uint64(2) {
		return 0, nil, errors.New("400 seqNo requested too far in future")
	}
	// A 3x target duration timeout is recommended for blocking requests, after which the server should return 503.
	d := time.Now().Add(mediaPlaylist.TargetDuration * 3)
	ctx, cancel := context.WithDeadline(context.Background(), d)
	defer cancel()
	start := time.Now()
	for {
		select {
		case <-ctx.Done():
			return time.Since(start), nil, errors.New("503 timeout")
		case _, ok := <-watcher.Events:
			if !ok {
				return time.Since(start), nil, errors.New("!ok from watcher")
			}
			mediaPlaylist, err := getMediaPlaylist(file)
			if err != nil {
				return time.Since(start), nil, err
			}
			if mediaPlaylist.LastMSN() > seqNo || (mediaPlaylist.LastMSN() == seqNo && mediaPlaylist.LastPart() >= partNo) {
				return time.Since(start), mediaPlaylist, nil
			}
		case err, ok := <-watcher.Errors:
			if !ok {
				return time.Since(start), nil, errors.New("!ok from watcher <-Watcher.Errors")
			}
			return time.Since(start), nil, err
		}
	}
}

func getReportFor(current, target string) string {
	file := target + "/" + index
	_, err := os.Stat(file)
	if err != nil {
		return ""
	}
	mediaPlaylist, err := getMediaPlaylist(file)
	if err != nil {
		return ""
	}
	p := mediaPlaylist.LastPart()
	m := mediaPlaylist.LastMSN()
	topLevelPath := filepath.Dir("/" + current + "/../../..") // current is the current lowLatencyHLS.m3u8 path
	uriString := filepath.Clean(fmt.Sprintf("%s/%s/%s", topLevelPath, target, playListEndPoint))
	return fmt.Sprintf("#EXT-X-RENDITION-REPORT:URI=\"%s\",LAST-MSN=%d,LAST-PART=%d\n", uriString, m, p)
}

func sendError(w http.ResponseWriter, r *http.Request, err error, status int, l logValues) {
	l.StatusCode = status
	if l.TotalDuration == 0 {
		l.TotalDuration = time.Since(l.StartTime)
	}
	log.Println(err)
	logLine(l)
	w.Header().Set("access-control-allow-origin", "*")
	w.Header().Set("access-control-expose-headers", "age")
	w.Header().Set("access-control-allow-headers", "Range")
	w.WriteHeader(int(status))
}

func logLine(l logValues) {
	fmt.Printf("%s %s %s %s %s %s %s %s %s %d %d %s\n",
		l.StartTime, l.Client, l.Protocol, l.Method, l.Scheme, l.HostHdr, l.RequestURI, l.BlockDuration, l.TotalDuration, l.Size, l.StatusCode, http.StatusText(int(l.StatusCode)))
}

func addHeaders(w http.ResponseWriter, file string, maxAge int, length int, blockDuration time.Duration) {
	if strings.HasSuffix(file, "mp4") {
		w.Header().Set("content-type", "video/mp4")
	} else if strings.HasSuffix(file, ".ts") {
		w.Header().Set("content-type", "video/mp2t")
	} else if strings.HasSuffix(file, ".m3u8") {
		w.Header().Set("content-type", "application/vnd.apple.mpegurl")
	}
	w.Header().Set("cache-control", fmt.Sprintf("max-age=%d", maxAge))
	if length != -1 {
		w.Header().Set("content-length", fmt.Sprintf("%d", length))
	}
	w.Header().Set("server", serverVersionString)
	w.Header().Set("block-duration", blockDuration.String())
	w.Header().Set("access-control-allow-origin", "*")
	w.Header().Set("access-control-expose-headers", "age")
	w.Header().Set("access-control-allow-headers", "Range")
}

func handler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	defer r.Body.Close()
	logV := logValues{StartTime: start, Client: r.RemoteAddr, Method: r.Method, Scheme: r.URL.Scheme, Protocol: r.Proto, RequestURI: r.URL.RequestURI(), HostHdr: r.Host}
	path := r.URL.EscapedPath()

	maxAge := 1
	if strings.HasSuffix(path, playListEndPoint) {
		seqParam := r.FormValue(seqParamQName)
		partParam := r.FormValue(partParamQName)
		skipParam := r.FormValue(skipParamQName)
		path = strings.TrimPrefix(path, "/")
		file := filepath.Dir(path) + "/" + index
		content := ""
		var currentMediaPlaylist *SimpleMediaPlaylist

		var seqNo, partNo uint64
		var err error
		blockingRequest := false
		if seqParam != "" {
			if seqNo, err = strconv.ParseUint(seqParam, 10, 64); err != nil {
				sendError(w, r, err, http.StatusBadRequest, logV)
				return
			}
			if partNo, err = strconv.ParseUint(partParam, 10, 64); err != nil {
				sendError(w, r, err, http.StatusBadRequest, logV)
				return
			}
			blockingRequest = true
		}

		var bDuration time.Duration
		bDuration, currentMediaPlaylist, err = waitForPlaylistWithSequenceNumber(file, seqNo, partNo)
		logV.BlockDuration = bDuration

		if blockingRequest {
			maxAge = 6 * int(currentMediaPlaylist.TargetDuration/time.Second)
		}
		if err != nil {
			sendError(w, r, err, http.StatusBadRequest, logV)
			return
		}
		if skipParam == "YES" {
			content = currentMediaPlaylist.EncodeWithSkip(canSkipUntil)
		} else {
			content = currentMediaPlaylist.Encode()
		}

		// add reports for directories that have the index file
		dirs, err := ioutil.ReadDir(".")
		if err != nil {
			sendError(w, r, err, http.StatusInternalServerError, logV)
			return
		}
		for _, p := range dirs {
			if strings.Contains("/"+path, p.Name()) {
				continue
			}
			content += getReportFor(path, p.Name())
		}

		content += "#\n"
		addHeaders(w, "file.m3u8", maxAge, len(content), bDuration)
		logV.Size = uint64(len(content))
		fmt.Fprint(w, content)
	} else {
		var bDuration time.Duration
		file := strings.TrimPrefix(path, "/")
		maxAge := 300 // This will be changed for the segment endpoint, default for static files (full segments)
		if strings.Contains(path, "/"+segmentEndPoint+".") {
			// First, check to see if this segment (part or full) was already listed
			indexFile := strings.TrimPrefix(filepath.Dir(path), "/") + "/" + index
			segURI := r.FormValue("segment")
			_, currentMediaPlaylist, err := waitForPlaylistWithSequenceNumber(indexFile, 0, 0)
			file = strings.TrimPrefix(filepath.Dir(path), "/") + "/" + segURI
			if err != nil {
				sendError(w, r, err, http.StatusBadRequest, logV)
				return
			}
			segmentReady := false
			for _, segment := range currentMediaPlaylist.Segments {
				for _, partial := range segment.Parts {
					if partial.URI == segURI {
						segmentReady = true
						break
					}
				}
			}

			maxAge = 6 * int(currentMediaPlaylist.TargetDuration/time.Second)

			if !segmentReady {
				// it was not listed yet, so now, wait for the next update of the playlist
				nextSeqNo := currentMediaPlaylist.LastMSN()
				nextPartNo := currentMediaPlaylist.LastPart()
				if nextPartNo == currentMediaPlaylist.MaxPartIndex {
					nextPartNo = 0
					nextSeqNo++
				} else {
					nextPartNo++
				}
				var err error
				bDuration, _, err = waitForPlaylistWithSequenceNumber(indexFile, nextSeqNo, nextPartNo)
				if err != nil {
					sendError(w, r, err, http.StatusBadRequest, logV)
					return
				}
			}
		}

		logV.BlockDuration = bDuration
		var err error
		var content []byte
		content, err = ioutil.ReadFile(file)
		if err != nil {
			sendError(w, r, err, http.StatusInternalServerError, logV)
			return
		}

		addHeaders(w, file, maxAge, -1, bDuration)
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, string(content))
		logV.Size = uint64(len(content))
	}
	logV.StatusCode = http.StatusOK
	logV.TotalDuration = time.Since(start)
	logLine(logV)
}

func main() {
	flag.Parse()

	if *dir != "" {
		err := os.Chdir(*dir)
		if err != nil {
			log.Fatalf("Can't cd to %s", *dir)
		}
	}

	http.HandleFunc("/", handler)
	if *certDir != "" {
		crtFile := *certDir + "/server.crt"
		keyFile := *certDir + "/server.key"
		fmt.Printf("Listening on https://%s/\n", *httpAddr)
		log.Fatalln(http.ListenAndServeTLS(*httpAddr, crtFile, keyFile, nil))
	} else {
		// for debugging only
		fmt.Printf("Listening on http://%s/\n", *httpAddr)
		log.Fatalln(http.ListenAndServe(*httpAddr, nil))
	}
}
