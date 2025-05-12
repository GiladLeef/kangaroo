import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), '../poolstats.json');
    
    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { 
          totalDP: 0, 
          clients: [] 
        }, 
        { status: 200 }
      );
    }
    
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(fileContent);
    
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Error reading pool stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch pool statistics' }, 
      { status: 500 }
    );
  }
} 