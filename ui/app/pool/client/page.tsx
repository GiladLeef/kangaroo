'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';

type ClientStats = {
  address: string;
  dpCount: number;
  percentage: number;
  lastSeen?: string;
  clientInfo?: string;
  totalDP: number;
};

export default function ClientStatsPage() {
  const searchParams = useSearchParams();
  const [address, setAddress] = useState<string>(searchParams.get('address') || '');
  const [stats, setStats] = useState<ClientStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    if (!address) {
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`/api/pool/client?address=${encodeURIComponent(address)}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch client statistics');
      }
      
      const data = await response.json();
      setStats(data);
    } catch (err) {
      setError('Error fetching client statistics. Please check your address and try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const addr = searchParams.get('address');
    if (addr) {
      setAddress(addr);
      
      // Set up automatic refresh
      fetchStats();
      const intervalId = setInterval(fetchStats, 30000);
      
      return () => clearInterval(intervalId);
    }
  }, [searchParams]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchStats();
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Client Statistics</h1>
          <Link href="/pool" className="text-blue-500 hover:underline">Back to Pool Dashboard</Link>
        </header>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Check Your Contribution</h2>
          <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-4">
            <input
              type="text"
              placeholder="Enter your Bitcoin address"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              className="flex-grow p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900"
            />
            <button
              type="submit"
              className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 md:whitespace-nowrap"
              disabled={loading || !address}
            >
              {loading ? 'Loading...' : 'Check Stats'}
            </button>
          </form>
        </div>

        {error && (
          <div className="bg-red-100 dark:bg-red-900 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-300 mb-8">
            {error}
          </div>
        )}

        {stats && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
              Statistics for {stats.address.substring(0, 10)}...{stats.address.substring(stats.address.length - 5)}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <div className="mb-6">
                  <h3 className="text-lg font-medium mb-2">Your Contribution</h3>
                  <div className="text-4xl font-bold">
                    {stats.dpCount.toLocaleString()}
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Distinguished Points
                  </p>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Pool Share</h3>
                  <div className="text-4xl font-bold">
                    {stats.percentage.toFixed(2)}%
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    of total pool contribution
                  </p>
                </div>
              </div>
              
              <div>
                <div className="mb-6">
                  <h3 className="text-lg font-medium mb-2">Pool Total</h3>
                  <div className="text-4xl font-bold">
                    {stats.totalDP.toLocaleString()}
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Distinguished Points
                  </p>
                </div>
                
                {stats.lastSeen && (
                  <div>
                    <h3 className="text-lg font-medium mb-2">Last Seen</h3>
                    <div className="text-xl">
                      {stats.lastSeen}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {stats.clientInfo}
                    </p>
                  </div>
                )}
              </div>
            </div>
            
            <div className="mt-8 pt-4 border-t border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-medium mb-3">Progress Visualization</h3>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                <div 
                  className="bg-blue-500 h-4 rounded-full"
                  style={{ width: `${Math.min(stats.percentage, 100)}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                Your contribution relative to the entire pool
              </p>
            </div>
            
            <div className="mt-8">
              <button
                onClick={fetchStats}
                className="text-blue-500 hover:underline"
              >
                Refresh Stats
              </button>
            </div>
          </div>
        )}
        
        {!stats && !error && !loading && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
            <p className="text-gray-600 dark:text-gray-300">
              Enter your Bitcoin address to check your contribution statistics.
            </p>
          </div>
        )}
      </div>
    </div>
  );
} 