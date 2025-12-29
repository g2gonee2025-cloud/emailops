
import { render, screen } from '@/tests/testUtils'; // Use custom render
import {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/Card';
import { describe, test, expect } from 'vitest';

describe('Card component and its parts', () => {
  test('renders Card with children', () => {
    render(<Card><div>Child content</div></Card>);
    expect(screen.getByText('Child content')).toBeInTheDocument();
  });

  test('renders CardHeader with children', () => {
    render(<CardHeader><div>Header content</div></CardHeader>);
    expect(screen.getByText('Header content')).toBeInTheDocument();
  });

  test('renders CardTitle with children', () => {
    render(<CardTitle>Title</CardTitle>);
    const titleElement = screen.getByText('Title');
    expect(titleElement).toBeInTheDocument();
    expect(titleElement.tagName).toBe('H3');
  });

  test('renders CardDescription with children', () => {
    render(<CardDescription>Description</CardDescription>);
    const descriptionElement = screen.getByText('Description');
    expect(descriptionElement).toBeInTheDocument();
    expect(descriptionElement.tagName).toBe('P');
  });

  test('renders CardContent with children', () => {
    render(<CardContent><div>Main content</div></CardContent>);
    expect(screen.getByText('Main content')).toBeInTheDocument();
  });

  test('renders CardFooter with children', () => {
    render(<CardFooter><div>Footer content</div></CardFooter>);
    expect(screen.getByText('Footer content')).toBeInTheDocument();
  });

  test('renders a full card structure', () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Test Title</CardTitle>
          <CardDescription>Test Description</CardDescription>
        </CardHeader>
        <CardContent>
          <p>This is the main content.</p>
        </CardContent>
        <CardFooter>
          <p>Footer information</p>
        </CardFooter>
      </Card>
    );

    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
    expect(screen.getByText('This is the main content.')).toBeInTheDocument();
    expect(screen.getByText('Footer information')).toBeInTheDocument();
  });
});
