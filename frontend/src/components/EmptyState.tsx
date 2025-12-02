export function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] p-6">
      <svg
        className="w-24 h-24 text-gray-300 mb-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
        />
      </svg>
      <h3 className="text-xl font-semibold text-gray-700 mb-2">No Devices Connected</h3>
      <p className="text-gray-500 text-center max-w-md">
        Waiting for devices to connect and send detection data. Make sure your detection systems
        are running and configured to connect to this dashboard.
      </p>
    </div>
  );
}
