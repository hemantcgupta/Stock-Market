import React, { Component } from 'react';
import cookieStorage from '../Constants/cookie-storage';
import api from '../API/api';
import eventApi from '../API/eventApi';
import $ from 'jquery';
import Sidebar from './Sidebar';
import Link from '@material-ui/core/Link';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import Divider from '@material-ui/core/Divider';
import FiberManualRecordIcon from '@material-ui/icons/FiberManualRecord';
import SupervisorAccountIcon from '@material-ui/icons/SupervisorAccount';
import PersonIcon from '@material-ui/icons/Person';
import VpnKeyIcon from '@material-ui/icons/VpnKey';
import LockIcon from '@material-ui/icons/Lock';
import IconButton from '@material-ui/core/IconButton';
import CloseIcon from '@material-ui/icons/Close';
import './component.scss'
import config from '../Constants/config'


class Headernav extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            userDetails: '',
            initial: '',
            menuData: [],
            toggleList: false
        };
    }

    componentWillMount() {
        const user = cookieStorage.getCookie('userDetails');
        this.setState({ userDetails: JSON.parse(user) }, () => this.getUserInitial())
        this.getMenuList();
    }

    getUserInitial() {
        const userName = this.state.userDetails.username;
        const initial = userName.charAt(0);
        this.setState({ initial })
    }

    sidemenuToggle = () => {

        if ($('body').hasClass('nav-md')) {
            $('#sidebar-menu').find('li.active ul').hide();
            $('#sidebar-menu').find('li.active').addClass('active-sm').removeClass('active');
        } else {
            $('#sidebar-menu').find('li.active-sm ul').show();
            $('#sidebar-menu').find('li.active-sm').addClass('active').removeClass('active-sm');
        }

        $('body').toggleClass('nav-md nav-sm');

    };

    _logout = () => {
        var eventData = {
            event_id: "2",
            description: "User logged out",
            where_path: "/",
            page_name: "Login Page"
        }
        eventApi.sendEvent(eventData)
        cookieStorage.deleteCookie('Authorization');
        cookieStorage.deleteCookie('Authorization');
        cookieStorage.deleteCookie('userDetails');
    }

    toggleList() {
        this.setState({ toggleList: !this.state.toggleList })
    }

    getMenuList = () => {
        api.get(`menus`)
            .then((response) => {
                if (response) {
                    if (response.data.length > 0) {
                        this.setState({ menuData: response.data });
                    }
                }
            })
            .catch((err) => {
                console.log('Menu Error', err);
            })
    }

    render() {
        return (
            <nav className="navbar navbar-default navbar-fixed-top">
                <div className="container">
                    <div className="navbar-header side-bar">
                        <Sidebar menuData={this.state.menuData} />
                        <a href="/posDashboard">ReveMax</a>
                    </div>
                    <div id="navbar" className="navbar-collapse collapse">
                        <ul className="nav navbar-nav navbar-right">
                            <IconButton
                                style={{ padding: this.state.toggleList ? '11px' : '1px' }}
                                aria-label="open drawer"
                                onClick={() => this.toggleList()}
                                edge="start"
                            >
                                {this.state.toggleList ? <CloseIcon className='close-btn' /> :
                                    <div className='initial-container'>
                                        <h3 className='initial'>{this.state.initial}</h3>
                                        <FiberManualRecordIcon className='circle' />
                                    </div>}
                            </IconButton>
                            {this.state.toggleList ?
                                <div className={`profile-dropdown`}>
                                    <List component="nav" aria-label="main mailbox folders">

                                        {this.state.userDetails.role === 'Super Admin' ?
                                            <div>
                                                <ListItem button>
                                                    <ListItemIcon color="inherit">
                                                        <SupervisorAccountIcon />
                                                    </ListItemIcon>
                                                    <Link href="/searchUser">
                                                        {"Portal"}
                                                    </Link>
                                                </ListItem>
                                                <Divider />
                                            </div> : ''}

                                        <ListItem button>
                                            <ListItemIcon color="inherit">
                                                <PersonIcon />
                                            </ListItemIcon>
                                            <Link href="/userProfile">
                                                {"Profile"}
                                            </Link>
                                        </ListItem>
                                        <Divider />

                                        <ListItem button>
                                            <ListItemIcon color="inherit">
                                                <VpnKeyIcon />
                                            </ListItemIcon>
                                            <Link href="/changePassword">
                                                {"Change Password"}
                                            </Link>
                                        </ListItem>
                                        <Divider />

                                        <ListItem button>
                                            <ListItemIcon color="inherit">
                                                <LockIcon />
                                            </ListItemIcon>
                                            <a href='/' onClick={() => this._logout()}>{"Logout"}</a>
                                        </ListItem>

                                    </List>
                                </div> : <div />}
                        </ul>
                    </div>
                </div>
            </nav>


        );
    }
}

export default Headernav;