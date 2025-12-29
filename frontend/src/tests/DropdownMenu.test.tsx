
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../components/ui/DropdownMenu';

describe('DropdownMenu', () => {
  it('renders the trigger and opens the menu on click', async () => {
    render(
      <DropdownMenu>
        <DropdownMenuTrigger>Open</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Item 1</DropdownMenuItem>
          <DropdownMenuItem>Item 2</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );

    const trigger = screen.getByText('Open');
    expect(trigger).toBeInTheDocument();

    expect(screen.queryByText('Item 1')).not.toBeInTheDocument();

    await userEvent.click(trigger);

    const menu = await screen.findByRole('menu');
    expect(menu).toBeInTheDocument();

    expect(within(menu).getByText('Item 1')).toBeInTheDocument();
    expect(within(menu).getByText('Item 2')).toBeInTheDocument();
  });
});
