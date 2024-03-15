
import React from 'react'
import IconButton from '@material-ui/core/IconButton';
import Link from '@material-ui/core/Link';
import MenuIcon from '@material-ui/icons/Menu';
import Drawer from '@material-ui/core/Drawer';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import TreeView from '@material-ui/lab/TreeView';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import TreeItem from '@material-ui/lab/TreeItem';
import './component.scss'

class SideBar extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            selectedId: null,
            left: false,
        }
    }


    toggleDrawer = (open) => (event) => {
        if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
            return;
        }

        this.setState({ left: open });
    };

    list = () => (
        <div
            className='sideBar'
            role="presentation"

        >
            <h4>Menus</h4>
            <List>
                {this.props.menuData.length > 0 ? this.props.menuData.map((menu, i) =>
                    <TreeView
                        defaultCollapseIcon={<ExpandMoreIcon />}
                        defaultExpandIcon={<ChevronRightIcon />}
                    >
                        <TreeItem nodeId={menu.ID} label={menu.Name}>
                            {menu.submenu ? menu.submenu.map((subMenu, j) =>
                                subMenu.visibility === 'show' ? <TreeItem nodeId={subMenu.ID} label={subMenu.Name} onClick={() => this.onListItemClick(subMenu)} onKeyDown={this.toggleDrawer(false)} /> : null
                            ) : ''}
                        </TreeItem>
                    </TreeView>
                ) : <ListItem style={{ color: 'white', fontSize: '14px' }}>No Menus to show</ListItem>}
            </List>
        </div>
    );

    onListItemClick(data) {
        this.setState({ selectedId: data.ID },
            () => {
                if (this.state.selectedId === data.ID) {
                    this.props.history.push({
                        pathname: data.Path,
                        state: { data: this.props.userData }
                    });
                    this.toggleDrawer(false)
                }
            })
    }

    render() {
        return (
            <div>
                <React.Fragment key={'left'}>
                    {/* <Button onClick={this.toggleDrawer(anchor, true)}>{anchor}</Button> */}
                    <IconButton
                        className='sidebarIcon'
                        color="inherit"
                        aria-label="open drawer"
                        onClick={this.toggleDrawer(true)}
                        edge="start"
                    >
                        <MenuIcon />
                    </IconButton>
                    <Drawer anchor={'left'} open={this.state.left} onClose={this.toggleDrawer(false)}>
                        {this.list()}
                    </Drawer>
                </React.Fragment>

            </div >
        );
    }
}

export default SideBar;
