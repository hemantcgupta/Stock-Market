import React from 'react'
import InfiniteScroll from "react-infinite-scroll-component";
import IconButton from '@material-ui/core/IconButton';
// import NotificationsIcon from '@material-ui/icons/Notifications';
import MessageIcon from '@material-ui/icons/Message';
import Drawer from '@material-ui/core/Drawer';
import { images } from '../Constants/images'
import Constant from '../Constants/validator'
import api from '../API/api'
import APIServices from '../API/apiservices';
import './component.scss'

const apiServices = new APIServices();

class NotificationRightSidebar extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            selectedId: null,
            right: false,
            messageArray: [],
            count: 1,
            hasMore: true,
            responseData: [],
            msgCount: 0,
            msgCountVisible: false
        }
    }

    componentWillReceiveProps = (newProps) => {
        const { messageArray, msgResponseData, msgCount, msgCountVisible, hasMore } = newProps;
        this.setState({
            messageArray,
            msgResponseData,
            msgCount,
            msgCountVisible,
            hasMore
        })
    }

    toggleDrawer = (open) => (event) => {
        if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
            return;
        }
        this.setState({ right: open, msgCountVisible: false });
    };

    list = () => (
        <div
            className='messageSideBar'
            role="presentation"

        >
            <h4>Messages</h4>
            <div className='messages'>
                <InfiniteScroll
                    dataLength={this.state.messageArray.length}
                    next={this.props.fetchMoreData}
                    hasMore={this.state.hasMore}
                    loader={
                        <div className={`msg-loader`} style={{ height: this.state.messageArray.length == 0 ? '100%' : 'auto' }}>
                            {this.state.messageArray.length == 0 ? <span>No messages to show</span> :
                                <div className='loading'><span>Loading</span><img src={images.ellipsis_loader} alt='loader' /></div>}
                        </div>}
                    height={'calc(100vh - 70px)'}
                    endMessage={
                        <p style={{ textAlign: "center", color: 'white', fontSize: '14px' }}>
                            <b>You have seen it all</b>
                        </p>}
                >
                    {this.state.messageArray.map((ele, index) => (
                        <div className='message-box' key={index}>
                            <p className='information'>{`${ele.userName},  ${this.convertDateTime(ele.messageTime)}`}</p>
                            <p>{ele.messageText} </p>
                            <div class="triangle-left"></div>
                        </div>
                    ))}
                </InfiniteScroll>
            </div>
        </div>
    );

    convertDateTime = (dateTime) => {
        const date = new Date(`${dateTime}`)
        return date.toUTCString()
    }
    render() {
        const { msgCountVisible, msgCount } = this.state;
        const msg_Count = msgCount > 10 ? `9+` : msgCount
        return (
            <div>
                <React.Fragment key={'right'}>
                    {/* <Button onClick={this.toggleDrawer(anchor, true)}>{anchor}</Button> */}
                    <IconButton
                        className='messageIcon'
                        color="inherit"
                        aria-label="open drawer"
                        onClick={this.toggleDrawer(true)}
                        edge="start"
                    >
                        <div style={{ position: 'relative' }}>
                            {msgCountVisible ?
                                <div className='circle-message'>
                                    <span className='count'>{msg_Count}</span>
                                </div> : ''}
                            <MessageIcon />
                        </div>
                    </IconButton>
                    <Drawer anchor={'right'} open={this.state.right} onClose={this.toggleDrawer(false)}>
                        {this.list()}
                    </Drawer>
                </React.Fragment>

            </div >
        );
    }
}

export default NotificationRightSidebar;
