// What it does:
//
// This example opens a video capture device, then streams MJPEG from it.
// Once running point your browser to the hostname/port you passed in the
// command line (for example http://localhost:8080) and you should see
// the live video stream.
//
// How to run:
//
// mjpeg-streamer [camera ID] [host:port]
//
//		go get -u github.com/hybridgroup/mjpeg
// 		go run ./demo/mjpeg-streamer-auth/main.go 1 0.0.0.0:8080
//
// To configure the username and password for basic auth, use the CVUSER and CVPASS
// environment variables. See https://12factor.net/config
//
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/hybridgroup/mjpeg"
	"gocv.io/x/gocv"
)

var (
	deviceID int
	err      error
	webcam   *gocv.VideoCapture
	stream   *mjpeg.Stream
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("How to run:\n\tmjpeg-streamer [camera ID] [host:port]")
		return
	}

	// parse args
	deviceID, _ = strconv.Atoi(os.Args[1])
	host := os.Args[2]

	// open webcam
	webcam, err = gocv.VideoCaptureDevice(deviceID)
	if err != nil {
		fmt.Printf("error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	// create the mjpeg stream
	stream = mjpeg.NewStream()

	// start capturing
	go capture()

	fmt.Println("Capturing. Point your browser to " + host)

	if os.Getenv("CVUSER") != "" {
		// http with basic authentication
		http.Handle("/", auth(stream.ServeHTTP))
	} else {
		// http without authentication
		http.Handle("/", stream)
	}

	log.Fatal(http.ListenAndServe(host, nil))
}

func capture() {
	img := gocv.NewMat()
	defer img.Close()

	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("cannot read device %d\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		buf, _ := gocv.IMEncode(".jpg", img)
		stream.UpdateJPEG(buf)
	}
}

func auth(fn http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		user, pass, _ := r.BasicAuth()

		// Store config in the environment: https://12factor.net/config
		if user == os.Getenv("CVUSER") && pass == os.Getenv("CVPASS") {
			fn(w, r)
			return
		}

		w.Header().Set("WWW-Authenticate", "Basic realm=\"localhost\"")
		http.Error(w, "Unauthorized.", 401)
		return
	}
}
