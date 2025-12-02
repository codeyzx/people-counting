import { useEffect, useRef, useState } from 'react';

interface LivePreviewProps {
  deviceId: string;
  frameData: string | null;
  width?: number;
  height?: number;
  className?: string;
}

export function LivePreview({ 
  deviceId, 
  frameData, 
  width = 640, 
  height = 480,
  className = ''
}: LivePreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fps, setFps] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const lastFrameTimeRef = useRef<number>(Date.now());
  const frameCountRef = useRef<number>(0);
  const fpsIntervalRef = useRef<number | null>(null);

  // Calculate FPS
  useEffect(() => {
    fpsIntervalRef.current = window.setInterval(() => {
      const now = Date.now();
      const elapsed = (now - lastFrameTimeRef.current) / 1000;
      
      if (elapsed > 0) {
        const currentFps = frameCountRef.current / elapsed;
        setFps(Math.round(currentFps));
        
        // Reset counters
        frameCountRef.current = 0;
        lastFrameTimeRef.current = now;
      }
    }, 1000);

    return () => {
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
      }
    };
  }, []);

  // Render frame to canvas
  useEffect(() => {
    if (!frameData || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      console.error('Failed to get canvas context');
      setHasError(true);
      return;
    }

    try {
      setHasError(false);
      
      // Create image from base64 data
      const img = new Image();
      
      img.onload = () => {
        try {
          // Draw image to canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          
          // Update frame counter
          frameCountRef.current++;
          setIsLoading(false);
        } catch (error) {
          console.error('Failed to draw image:', error);
          setHasError(true);
        }
      };
      
      img.onerror = () => {
        console.error('Failed to load image');
        setHasError(true);
        setIsLoading(false);
      };
      
      // Set image source
      img.src = `data:image/jpeg;base64,${frameData}`;
      
    } catch (error) {
      console.error('Failed to decode frame:', error);
      setHasError(true);
      setIsLoading(false);
    }
  }, [frameData]);

  return (
    <div className={`relative ${className}`}>
      {/* Canvas for video display */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-auto bg-gray-900 rounded-lg"
      />
      
      {/* Loading state */}
      {isLoading && !hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 rounded-lg">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-2"></div>
            <p className="text-gray-300 text-sm">Waiting for video...</p>
          </div>
        </div>
      )}
      
      {/* Error state */}
      {hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 rounded-lg">
          <div className="text-center">
            <svg className="w-12 h-12 text-red-500 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-gray-300 text-sm">Failed to load video</p>
          </div>
        </div>
      )}
      
      {/* FPS counter */}
      {!isLoading && !hasError && (
        <div className="absolute top-2 right-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
          {fps} FPS
        </div>
      )}
      
      {/* Device ID label */}
      <div className="absolute bottom-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
        {deviceId}
      </div>
    </div>
  );
}
