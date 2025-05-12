import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request: NextRequest) {
  try {
    // Get the bitcoin address from the query string
    const { searchParams } = new URL(request.url);
    const address = searchParams.get('address');
    
    if (!address) {
      return NextResponse.json(
        { error: 'Bitcoin address is required' }, 
        { status: 400 }
      );
    }
    
    const filePath = path.join(process.cwd(), '../poolstats.json');
    
    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { 
          address,
          dpCount: 0,
          percentage: 0,
          totalDP: 0 
        }, 
        { status: 200 }
      );
    }
    
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(fileContent);
    
    // Find client by address
    const client = data.clients.find((c: any) => c.address === address);
    
    if (!client) {
      return NextResponse.json(
        { 
          address,
          dpCount: 0,
          percentage: 0,
          totalDP: data.totalDP,
          expectedDP: data.expectedDP
        }, 
        { status: 200 }
      );
    }
    
    return NextResponse.json({
      address: client.address,
      dpCount: client.dpCount,
      percentage: client.percentage,
      lastSeen: client.lastSeen,
      clientInfo: client.clientInfo,
      totalDP: data.totalDP,
      expectedDP: data.expectedDP
    }, { status: 200 });
  } catch (error) {
    console.error('Error reading pool stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch client statistics' }, 
      { status: 500 }
    );
  }
} 