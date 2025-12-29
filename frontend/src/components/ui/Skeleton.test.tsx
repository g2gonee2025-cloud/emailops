import { render } from '@testing-library/react';
import * as React from 'react';
import { Skeleton } from './Skeleton';

describe('Skeleton', () => {
  it('should forward a ref to the underlying div element', () => {
    const ref = React.createRef<HTMLDivElement>();
    render(<Skeleton ref={ref} />);
    expect(ref.current).not.toBeNull();
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });
});
