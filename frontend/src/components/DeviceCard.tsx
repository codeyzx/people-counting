import type { DeviceInfo } from '../types';
import { StatusBadge } from './StatusBadge';
import { CrowdStatusBadge } from './CrowdStatusBadge';
import { LivePreview } from './LivePreview';
import { formatRelativeTime } from '../utils';

interface DeviceCardProps {
  device: DeviceInfo;
  frameData?: string | null;
}

export function DeviceCard({ device, frameData }: DeviceCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-200">
      {/* Live Preview */}
      {frameData && (
        <div className="w-full">
          <LivePreview
            deviceId={device.source_id}
            frameData={frameData}
            width={640}
            height={480}
          />
        </div>
      )}
      
      {/* No video placeholder */}
      {!frameData && (
        <div className="w-full h-64 bg-gray-900 flex items-center justify-center">
          <div className="text-center">
            <svg className="w-16 h-16 text-gray-600 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <p className="text-gray-500 text-sm">No video stream</p>
          </div>
        </div>
      )}
      
      {/* Device Info */}
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">{device.source_id}</h3>
          <StatusBadge status={device.status} />
        </div>

        {/* Crowd Count */}
        <div className="mb-4">
          <div className="flex items-center mb-2">
            <svg
              className="w-5 h-5 text-gray-500 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
              />
            </svg>
            <span className="text-sm text-gray-600">Crowd Count</span>
          </div>
          <p className="text-3xl font-bold text-gray-900">{device.current_count}</p>
        </div>

        {/* Crowd Status */}
        <div className="mb-4">
          <CrowdStatusBadge status={device.crowd_status} count={device.current_count} />
        </div>

        {/* Last Seen */}
        <div className="flex items-center text-sm text-gray-500">
          <svg
            className="w-4 h-4 mr-1.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span>Last seen {formatRelativeTime(device.last_seen)}</span>
        </div>

        {/* Frame Number (optional metadata) */}
        <div className="mt-2 text-xs text-gray-400">
          Frame: {device.frame_number}
        </div>
      </div>
    </div>
  );
}
